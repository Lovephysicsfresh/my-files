"""第八步：候选筛选 —— 论文真正在做的事

前面我们一直在练"训练模型"。这一步换成论文的主线：
给定一堆候选组成，用模型筛出值得做实验的那些。

流程（完全按论文）：
    1. 从作者的候选表读组成（OER_screen_initial.csv）
    2. 按 SI 规则把金属随机放进超胞位点
         A 位某元素原子数 = round(36 × 该元素在 A 位的分数)
         B 位某元素原子数 = round(60 × 该元素在 B 位的分数)
    3. 用作者的 5 个折模型各预测一次
    4. 均值 = 预测过电位，标准差 = 不确定度
    5. 统计 σ < 0.020 V 的比例

关于 XRD：候选没有实验 XRD，作者的生成模型权重也没公开（仓库里是空目录）。
我们测过：把 XRD 全置零，预测只偏 0.02 V 左右（0.5252 vs 0.5463），
所以这里用零向量代替，并在结果里注明。

运行：python step_17_candidate_screen.py
耗时：约 7 分钟（50 个候选 × 5 个模型）

改数量：$env:NCAND = "20"
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import (
    ELECTRONEGATIVITY,
    IONIC_RADIUS,
    GRAPH_DIR,
    build_dense_variant,
    parse_pdb,
)
from step_04_load_checkpoint import Net

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
CKPT = ROOT / "OER2022" / "initial_train" / "checkpoint"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

NCAND = int(os.environ.get("NCAND", "50"))
A_ELEMENTS = ["La", "Ba", "Sr", "Pr", "Ca"]
B_ELEMENTS = ["Mn", "Fe", "Co", "Ni", "Cu"]
THRESHOLD = 0.020
N_A, N_B = 36, 60


def apportion(weights, total):
    """把 total 个位点按比例分给各元素，用最大余数法保证加起来正好是 total。"""
    raw = weights * total
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for i in order[:remainder]:
            base[i] += 1
    elif remainder < 0:
        order = np.argsort(raw - base)
        for i in order[:abs(remainder)]:
            base[i] -= 1
    return base


def build_nodes(template, a_idx, b_idx, x_a, x_b):
    """按组成生成 336 行节点特征。"""
    n_a = apportion(x_a, N_A)
    n_b = apportion(x_b, N_B)

    assign = {}
    pos = 0
    for el, cnt in zip(A_ELEMENTS, n_a):
        for _ in range(cnt):
            assign[a_idx[pos]] = el
            pos += 1
    pos = 0
    for el, cnt in zip(B_ELEMENTS, n_b):
        for _ in range(cnt):
            assign[b_idx[pos]] = el
            pos += 1

    rows = []
    for i, atom in enumerate(template):
        el = assign.get(i)
        if el is None:
            rows.append([IONIC_RADIUS["O"], ELECTRONEGATIVITY["O"], 0.0, 0.0])
        else:
            is_a = 1.0 if atom["site"] == "A" else 0.0
            is_b = 1.0 if atom["site"] == "B" else 0.0
            rows.append([IONIC_RADIUS[el], ELECTRONEGATIVITY[el], is_a, is_b])
    return np.asarray(rows, np.float32), n_a, n_b


def formula(x_a, x_b):
    def part(elems, w):
        s = [f"{el}{w[i]:.1f}" for i, el in enumerate(elems) if w[i] > 1e-6]
        return "".join(s)
    return f"{part(A_ELEMENTS, x_a)}{part(B_ELEMENTS, x_b)}O3"


def main():
    print("=" * 72)
    print("第八步：候选筛选（用作者的 5 个折模型）")
    print("=" * 72)

    template, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    a_idx = [i for i, a in enumerate(template) if a["site"] == "A"]
    b_idx = [i for i, a in enumerate(template) if a["site"] == "B"]
    print(f"  模板：A 位 {len(a_idx)} 个，B 位 {len(b_idx)} 个，"
          f"氧 {len(template) - len(a_idx) - len(b_idx)} 个")

    # 拓扑固定，只算一次
    pairs, edges = build_dense_variant(template, conect)
    pairs_tf = tf.constant(pairs, tf.int32)
    edges_tf = tf.constant(edges, tf.float32)
    print(f"  边：{edges.shape[0]} 条（全组合）")

    cand = pd.read_csv(ROOT / "OER2022" / "initial_screen" / "OER_screen_initial.csv")
    cand = cand.head(NCAND)
    print(f"  候选：{len(cand)} 个（表里共 10131 行）")

    nets = []
    n_models = int(os.environ.get("NMODELS", "20"))
    for k in range(n_models):
        path = CKPT / f"OO_{k:02d}_full"
        if (Path(str(path) + ".index")).exists():
            nets.append(Net(path).load())
    print(f"  模型：{len(nets)} 个全量模型（OO_00_full ~ OO_{len(nets) - 1:02d}_full）")
    print("        论文筛选用的正是 num_seed=20 的全量模型，不是折模型")

    zero7 = np.zeros(7, np.float32)
    records = []

    for n, row in enumerate(cand.itertuples(), 1):
        x_a = np.array([getattr(row, f"atom_{el}") for el in A_ELEMENTS], float)
        x_b = np.array([getattr(row, f"atom_{el}") for el in B_ELEMENTS], float)
        tsne = np.array([row.tsne_0, row.tsne_1], np.float32)

        nodes, n_a, n_b = build_nodes(template, a_idx, b_idx, x_a, x_b)

        preds = []
        for net in nets:
            p = net.predict(nodes, pairs_tf, edges_tf, zero7, zero7, tsne)
            preds.append(p)
        preds = np.asarray(preds)

        records.append({
            "id": row.id,
            "formula": formula(x_a, x_b),
            "n_metals": int((x_a > 1e-6).sum() + (x_b > 1e-6).sum()),
            "mean": float(preds.mean()),
            "std": float(preds.std(ddof=1)),
            "single": " ".join(f"{p:.3f}" for p in preds),
        })

        if n % 10 == 0 or n == len(cand):
            print(f"  已完成 {n}/{len(cand)}   "
                  f"最近一个 {records[-1]['formula']}: "
                  f"均值 {records[-1]['mean']:.3f} ± {records[-1]['std']:.3f}")

    df = pd.DataFrame(records)
    df = df.sort_values("mean").reset_index(drop=True)
    df.to_csv(RES / "step17_candidates.csv", index=False, encoding="utf-8-sig")

    pass_rate = float((df["std"] < THRESHOLD).mean())
    print()
    print("=" * 72)
    print("结果")
    print("=" * 72)
    print(f"  候选数                 : {len(df)}")
    print(f"  模型数                 : {len(nets)}")
    print(f"  平均不确定度            : {df['std'].mean():.4f} V")
    print(f"  不确定度中位数          : {df['std'].median():.4f} V")
    print(f"  σ < {THRESHOLD} V 的比例  : {pass_rate * 100:.2f}%")
    print(f"  论文 OO-1 的通过率       : 2.87%（10101 个候选）")
    print()
    print("  预测最好的 10 个：")
    print(f"  {'组成':<28}{'金属数':>6}{'预测η':>9}{'不确定度':>10}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['formula']:<28}{r['n_metals']:>6}{r['mean']:>9.3f}{r['std']:>10.4f}")

    # 图：模仿论文 Fig 3a
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ok = df["std"] < THRESHOLD
    ax.scatter(df.loc[~ok, "mean"], df.loc[~ok, "std"], s=18, c="#bbbbbb",
               label=f"outside ({100 - pass_rate * 100:.1f}%)")
    ax.scatter(df.loc[ok, "mean"], df.loc[ok, "std"], s=26, c="#d62728",
               label=f"inside ({pass_rate * 100:.1f}%)")
    ax.axhline(THRESHOLD, color="k", ls="--", lw=1, label=f"threshold = {THRESHOLD} V")
    ax.set_xlabel("predicted overpotential (V)")
    ax.set_ylabel("uncertainty (V)")
    ax.set_title(f"Candidate screening with {len(nets)} models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step17_screen.png", dpi=150)
    plt.close(fig)

    print()
    print(f"  图已保存: {FIG / 'step17_screen.png'}")
    print(f"  表已保存: {RES / 'step17_candidates.csv'}")
    print()
    print("  说明：候选没有实验 XRD，这里用零向量代替（实测只偏约 0.02 V）。")
    print("        作者的生成模型权重未公开，所以这一步无法完全照原样做。")


if __name__ == "__main__":
    main()

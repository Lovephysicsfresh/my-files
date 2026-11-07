"""第六步：集成（bootstrap ensemble）—— 论文最核心的方法

前面几步已经证明：
  · 单个模型的 corr 在 0.3-0.8 之间剧烈摆动
  · 换个随机种子，整个模型的预测会整体平移
  · dropout 能推迟过拟合，但救不了单个模型的不稳定

这一步做论文真正做的事：
  · 同一折，训练 N 个模型（不同随机种子）
  · 每个模型给出自己的一套预测
  · 把它们平均

你要看的是这个对比：
  N 个模型各自的 corr   vs   平均之后的 corr

同时输出每个样品的标准差 —— 这就是论文里的"不确定度"。

运行：python step_14_ensemble.py
耗时：约 37 分钟（5 个种子 × 120 轮）

改数量：$env:NSEEDS = "3"   改轮数：$env:EPOCHS = "200"
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from step_10_train_from_scratch import BATCH, TrainableMPNN, pearson
from step_11_fivefold import load_all, pack

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", "120"))
NSEEDS = int(os.environ.get("NSEEDS", "5"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))


def train_model(Xtr, ytr, seed):
    tf.keras.utils.set_random_seed(seed)
    model = TrainableMPNN(dropout=DROPOUT)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    n = len(ytr)
    for epoch in range(EPOCHS):
        order = np.random.RandomState(seed * 7919 + epoch).permutation(n)
        for start in range(0, n, BATCH):
            b = order[start:start + BATCH]
            batch = [tf.constant(x.numpy()[b]) for x in Xtr]
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.constant(ytr[b]), model(batch, training=True))
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
    return model


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    print("=" * 70)
    print(f"集成实验：{NSEEDS} 个模型 × {EPOCHS} 轮，dropout={DROPOUT}")
    print("=" * 70)

    nodes, pairs, edges, feat, y = load_all(ids)
    val_idx = np.arange(0, 5)
    tr_idx = np.arange(5, 30)
    print(f"  训练 {len(tr_idx)} 个，验证 {len(val_idx)} 个 ({', '.join(ids[:5])})")

    Xtr = pack(nodes, pairs, edges, feat, tr_idx)
    ytr = y[tr_idx]
    Xv = pack(nodes, pairs, edges, feat, val_idx)
    yv = y[val_idx]

    preds = []
    for k in range(NSEEDS):
        seed = k
        print(f"\n  训练第 {k + 1}/{NSEEDS} 个模型 (seed={seed}) ...", end="", flush=True)
        model = train_model(Xtr, ytr, seed)
        p = model(Xv, training=False).numpy()
        preds.append(p)
        print(f" 完成, 该模型 corr = {pearson(yv, p):.4f}")

    preds = np.stack(preds)                 # (NSEEDS, 5)
    ens = preds.mean(axis=0)                # 集成预测
    unc = preds.std(axis=0, ddof=1)         # 集成不确定度

    indiv_corr = [pearson(yv, preds[i]) for i in range(NSEEDS)]
    ens_corr = pearson(yv, ens)
    indiv_rmse = [float(np.sqrt(np.mean((yv - preds[i]) ** 2))) for i in range(NSEEDS)]
    ens_rmse = float(np.sqrt(np.mean((yv - ens) ** 2)))

    print()
    print("=" * 70)
    print("逐个模型的预测")
    print("=" * 70)
    head = f"{'样品':<8}{'实验值':>8}" + "".join(f"{'m' + str(i):>8}" for i in range(NSEEDS))
    print(head + f"{'集成':>8}{'标准差':>8}")
    for i, pid in enumerate(ids[:5]):
        row = f"{pid:<8}{yv[i]:>8.4f}" + "".join(f"{preds[j][i]:>8.3f}" for j in range(NSEEDS))
        row += f"{ens[i]:>8.4f}{unc[i]:>8.4f}"
        print(row)

    print()
    print("=" * 70)
    print("核心对比")
    print("=" * 70)
    print(f"{'模型':<12}{'corr':>10}{'RMSE(V)':>11}")
    for i in range(NSEEDS):
        print(f"{'单个 m' + str(i):<12}{indiv_corr[i]:>10.4f}{indiv_rmse[i]:>11.4f}")
    print("-" * 33)
    print(f"{'单模型平均':<12}{np.mean(indiv_corr):>10.4f}{np.mean(indiv_rmse):>11.4f}")
    print(f"{'集成（平均）':<12}{ens_corr:>10.4f}{ens_rmse:>11.4f}")
    print()
    print(f"  集成 vs 单模型最好:  {ens_corr:.4f} vs {max(indiv_corr):.4f}")
    print(f"  集成 vs 单模型平均:  {ens_corr:.4f} vs {np.mean(indiv_corr):.4f}")
    print(f"  提升:              {ens_corr - np.mean(indiv_corr):+.4f}")
    print()
    print(f"  平均不确定度: {unc.mean():.4f} V")
    print(f"  论文的判据是 0.020 V，本组中 < 0.020 的样品数: "
          f"{int((unc < 0.020).sum())}/{len(unc)}")

    # 图1：单模型 corr vs 集成 corr
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    xs = np.arange(NSEEDS)
    ax.bar(xs, indiv_corr, color="#bbbbbb", label="single models")
    ax.axhline(ens_corr, color="#d62728", lw=2, ls="--", label=f"ensemble = {ens_corr:.3f}")
    ax.axhline(np.mean(indiv_corr), color="#1f77b4", lw=1.5, ls=":", label=f"mean = {np.mean(indiv_corr):.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"m{i}" for i in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("validation correlation")
    ax.set_title("Averaging beats individual models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot([0.4, 0.65], [0.4, 0.65], "k--", lw=1, label="perfect")
    for i in range(NSEEDS):
        ax.scatter(yv, preds[i], s=22, color="#bbbbbb", alpha=0.8)
    ax.errorbar(yv, ens, yerr=unc, fmt="o", color="#d62728", capsize=4,
                label="ensemble", zorder=5)
    ax.set_xlabel("experimental overpotential (V)")
    ax.set_ylabel("predicted overpotential (V)")
    ax.set_title("Ensemble prediction with uncertainty")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "step14_ensemble.png", dpi=150)
    plt.close(fig)

    np.savetxt(ROOT / "results" / "step14_ensemble.csv",
               np.c_[yv, ens, unc], delimiter=",",
               header="experimental,ensemble_mean,ensemble_std", comments="")

    print()
    print("=" * 70)
    print("这一步你看到的是论文的什么")
    print("=" * 70)
    print("  1. 单个模型之间差异很大（看左图的柱子高度）")
    print("  2. 平均之后，误差互相抵消，corr 上升")
    print("  3. 各模型预测的分散程度 = 不确定度")
    print("  4. 论文用 10 个种子 × 5 折 = 50 个模型，我们这里只用了", NSEEDS, "个")
    print()
    print(f"  图已保存: {FIG / 'step14_ensemble.png'}")


if __name__ == "__main__":
    main()

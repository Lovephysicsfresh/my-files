"""第七步：换回论文真正的输入表示 —— 50400 条边

前面所有训练实验都用了 1296 条真实化学键，那是个妥协（为了速度）。
但用作者 checkpoint 做的判别实验已经证明：

    1296 条边  -> 预测 0.486
    50400 条边 -> 预测 0.546   （作者真值 0.5348）

也就是说 50400 条才是对的。这一步把输入换回来。

设计：成对跑，只改一个变量
    对照组  1296 条边   batch 1  60 轮  dropout 0.1  seed 0
    实验组  50400 条边  batch 1  60 轮  dropout 0.1  seed 0

batch 两边都是 1（50400 条边用 batch 5 会吃掉 4 GB 内存）。

耗时：对照组约 5 分钟，实验组约 2-4 小时（建议睡前跑）

运行：
    python step_16_dense_input.py

想缩短：$env:EPOCHS = "30"
只跑实验组：$env:VARIANTS = "dense"
"""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import (
    GRAPH_DIR,
    build_conect_variant,
    build_dense_variant,
    node_features,
    parse_pdb,
)
from step_10_train_from_scratch import TrainableMPNN, pearson

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", "60"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
BATCH = int(os.environ.get("BATCH", "1"))
SEED = int(os.environ.get("SEED", "0"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5"))
VARIANTS = [v.strip() for v in os.environ.get("VARIANTS", "sparse,dense").split(",")]

BUILDERS = {"sparse": build_conect_variant, "dense": build_dense_variant}


def load(ids, kind):
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    build = BUILDERS[kind]

    nodes, pairs, edges, inten, pos, tsne, y = [], [], [], [], [], [], []
    for pid in ids:
        row = df[df["id"] == pid].iloc[0]
        atoms, conect = parse_pdb(GRAPH_DIR / f"{pid}_00.pdb")
        p, e = build(atoms, conect)
        nodes.append(node_features(atoms))
        pairs.append(p)
        edges.append(e)
        inten.append(row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32))
        pos.append(row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32))
        tsne.append(row[["tsne_0", "tsne_1"]].to_numpy(np.float32))
        y.append(np.float32(row["potential"]))

    return ([np.stack(nodes).astype(np.float32),
             np.stack(pairs).astype(np.int32),
             np.stack(edges).astype(np.float32),
             np.stack(inten).astype(np.float32),
             np.stack(pos).astype(np.float32),
             np.stack(tsne).astype(np.float32)],
            np.asarray(y, np.float32))


def run(kind, data, y, tr_idx, val_idx):
    X, yv = data, y[val_idx]
    n_edges = int(X[2].shape[1])
    print()
    print("=" * 70)
    print(f"[{kind}] 每条图的边数 = {n_edges}   batch = {BATCH}   "
          f"epochs = {EPOCHS}   dropout = {DROPOUT}")
    print("=" * 70)

    tf.keras.utils.set_random_seed(SEED)
    model = TrainableMPNN(dropout=DROPOUT)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    def batch_of(idx):
        return [tf.constant(x[idx]) for x in X]

    ytr = y[tr_idx]

    def predict_each(idx):
        """逐个样本前向。50400 条边时批量送会 OOM，只能一个一个来。"""
        out = []
        for i in idx:
            out.append(float(model([tf.constant(x[[i]]) for x in X], training=False)[0]))
        return np.asarray(out, np.float32)

    def evaluate():
        p_tr = predict_each(tr_idx)
        mse_tr = float(np.mean((ytr - p_tr) ** 2))
        p_v = predict_each(val_idx)
        return mse_tr, pearson(yv, p_v), float(np.sqrt(np.mean((yv - p_v) ** 2)))

    csv_path = RES / f"step16_{kind}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_mse,val_corr,val_rmse\n")

    records = [(0, *evaluate())]
    print(f"  epoch   0/{EPOCHS}  train MSE {records[0][1]:.6f}  "
          f"val corr {records[0][2]:.4f}  val RMSE {records[0][3]:.4f}")

    t0 = time.time()
    n = len(tr_idx)
    for epoch in range(EPOCHS):
        order = np.random.RandomState(SEED * 7919 + epoch).permutation(n)
        for start in range(0, n, BATCH):
            b = order[start:start + BATCH]
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.constant(ytr[b]), model(batch_of(b), training=True))
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            rec = (epoch + 1, *evaluate())
            records.append(rec)
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{rec[0]},{rec[1]:.8f},{rec[2]:.8f},{rec[3]:.8f}\n")
            elapsed = time.time() - t0
            per_ep = elapsed / (epoch + 1)
            eta = per_ep * (EPOCHS - epoch - 1) / 60
            print(f"  epoch {epoch + 1:>3}/{EPOCHS}  train MSE {rec[1]:.6f}  "
                  f"val corr {rec[2]:.4f}  val RMSE {rec[3]:.4f}   "
                  f"已用 {elapsed / 60:.1f} 分，预计还要 {eta:.0f} 分")

    rec = np.asarray(records)
    p_final = predict_each(val_idx)
    np.savetxt(RES / f"step16_{kind}_pred.csv", np.c_[yv, p_final], delimiter=",",
               header="experimental,predicted", comments="")
    print(f"  [{kind}] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    return rec, p_final


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()
    val_idx = np.arange(0, 5)
    tr_idx = np.arange(5, 30)
    yv = df.set_index("id").loc[ids[:5], "potential"].to_numpy(np.float32)
    const_rmse = float(np.sqrt(np.mean((yv - yv.mean()) ** 2)))

    print("=" * 70)
    print("第七步：输入表示对照（1296 条边 vs 50400 条边）")
    print("=" * 70)
    print(f"  验证集: {', '.join(ids[:5])}")
    print(f"  参照：全部猜平均值的 RMSE = {const_rmse:.4f} V")

    curves, preds = {}, {}
    for kind in VARIANTS:
        data, y = load(ids, kind)
        curves[kind], preds[kind] = run(kind, data, y, tr_idx, val_idx)
        del data

    print()
    print("=" * 70)
    print("对照结果")
    print("=" * 70)
    print(f"{'输入':<16}{'边数':>9}{'训练RMSE':>11}{'验证corr':>10}{'验证RMSE':>11}")
    for kind, rec in curves.items():
        n_e = 1296 if kind == "sparse" else 50400
        print(f"{kind:<16}{n_e:>9}{np.sqrt(rec[-1, 1]):>11.4f}"
              f"{rec[-1, 2]:>10.4f}{rec[-1, 3]:>11.4f}")
    print(f"{'[基线] 猜平均':<16}{'-':>9}{'-':>11}{0.0:>10.4f}{const_rmse:>11.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = {"sparse": "#d62728", "dense": "#1f77b4"}
    labels = {"sparse": "edges 1296 (real bonds)", "dense": "edges 50400 (all pairs)"}
    for ax, col, title in [(axes[0], 1, "Train error"), (axes[1], 2, "Validation correlation")]:
        for kind, rec in curves.items():
            ax.plot(rec[:, 0], rec[:, col], "-o", ms=3,
                    color=colors.get(kind, "k"), label=labels.get(kind, kind))
        if col == 1:
            ax.set_yscale("log")
            ax.set_ylabel("training MSE (log)")
        else:
            ax.set_ylabel("validation correlation")
            ax.axhline(0, color="gray", lw=0.8, ls=":")
            ax.set_ylim(-1, 1)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step16_dense_input.png", dpi=150)
    plt.close(fig)

    print()
    print(f"  图已保存: {FIG / 'step16_dense_input.png'}")
    print()
    print("  判读：")
    print("    如果 dense 的验证 RMSE 明显低于 sparse，说明输入表示确实是主因，")
    print("    前面几轮调参都是在错误的地基上盖楼。")
    print("    如果两者差不多，说明 50400 条边不是关键，问题在别处（数据量、模型容量）。")


if __name__ == "__main__":
    main()

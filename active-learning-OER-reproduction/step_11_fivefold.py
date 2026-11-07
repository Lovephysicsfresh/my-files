"""第五步：完整的六折交叉验证 —— 让 30 个样品每个都被预测一次。

和第四步的区别：
  第四步：只跑了 1 折，只验证 5 个样品，corr 在 5 个点上非常不稳定
  第五步：6 折全跑，30 个样品都有预测，这样才算得出和论文可比的 corr

论文的做法（SI 第 3 节）：
  30 个样本 -> 分成 6 折，每折 5 个
  每次拿 5 折训练、1 折验证
  6 个模型都训完之后，每个样品恰好被预测 1 次

运行：python step_11_fivefold.py
耗时：约 22 分钟（6 折 x 60 轮）
"""

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
    node_features,
    parse_pdb,
)
from step_10_train_from_scratch import (
    BATCH,
    EPOCHS,
    SEEDS,
    TrainableMPNN,
    pearson,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

N_FOLDS = 6
SEED = SEEDS[0]


def load_all(ids):
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    cols = [f"intensity_{i}" for i in range(7)] + [f"position_{i}" for i in range(7)]

    nodes, pairs, edges, feat, y = [], [], [], [], []
    for pid in ids:
        row = df[df["id"] == pid].iloc[0]
        atoms, conect = parse_pdb(GRAPH_DIR / f"{pid}_00.pdb")
        p, e = build_conect_variant(atoms, conect)
        nodes.append(node_features(atoms))
        pairs.append(p)
        edges.append(e)
        feat.append(np.concatenate([
            row[cols[:7]].to_numpy(np.float32),
            row[cols[7:]].to_numpy(np.float32),
            row[["tsne_0", "tsne_1"]].to_numpy(np.float32),
        ]))
        y.append(np.float32(row["potential"]))

    return (
        np.stack(nodes).astype(np.float32),
        np.stack(pairs).astype(np.int32),
        np.stack(edges).astype(np.float32),
        np.stack(feat).astype(np.float32),
        np.asarray(y, np.float32),
    )


def pack(nodes, pairs, edges, feat, idx):
    return [
        tf.constant(nodes[idx]),
        tf.constant(pairs[idx]),
        tf.constant(edges[idx]),
        tf.constant(feat[idx][:, :7]),
        tf.constant(feat[idx][:, 7:14]),
        tf.constant(feat[idx][:, 14:]),
    ]


def train_fold(nodes, pairs, edges, feat, y, tr_idx, seed):
    tf.keras.utils.set_random_seed(seed)
    model = TrainableMPNN()
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    Xtr = pack(nodes, pairs, edges, feat, tr_idx)
    ytr = tf.constant(y[tr_idx])
    n = len(tr_idx)

    loss = 0.0
    for epoch in range(EPOCHS):
        order = np.random.RandomState(seed * 7919 + epoch).permutation(n)
        epoch_loss, nb = 0.0, 0
        for start in range(0, n, BATCH):
            b = order[start:start + BATCH]
            batch = [x.numpy()[b] for x in Xtr]
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.gather(ytr, b), model(batch))
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss += float(loss)
            nb += 1
        loss = epoch_loss / nb
        if (epoch + 1) % 20 == 0:
            print(f"        epoch {epoch + 1:>3}/{EPOCHS}  loss = {loss:.6f}")
    return model


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    print("=" * 70)
    print("载入 30 个样品的图和数据")
    print("=" * 70)
    nodes, pairs, edges, feat, y = load_all(ids)
    print(f"  nodes {nodes.shape}   pairs {pairs.shape}   edges {edges.shape}")
    print(f"  30 个样品的实验过电位: {y.min():.3f} - {y.max():.3f} V")

    folds = [np.arange(i * 5, (i + 1) * 5) for i in range(N_FOLDS)]
    pred = np.full(len(ids), np.nan)

    print()
    print("=" * 70)
    print(f"开始 {N_FOLDS} 折交叉验证（每折 {EPOCHS} 轮，共约 {N_FOLDS * EPOCHS * 3.7 / 60:.0f} 分钟）")
    print("=" * 70)

    for k, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(ids)), val_idx)
        print(f"\n  第 {k + 1} 折   训练 {len(tr_idx)} 个，验证 {[ids[i] for i in val_idx]}")
        model = train_fold(nodes, pairs, edges, feat, y, tr_idx, seed=SEED + k)
        Xv = pack(nodes, pairs, edges, feat, val_idx)
        pred[val_idx] = model(Xv).numpy()

    ok = ~np.isnan(pred)
    yv, pv = y[ok], pred[ok]
    corr = pearson(yv, pv)
    rmse = float(np.sqrt(np.mean((yv - pv) ** 2)))
    mae = float(np.mean(np.abs(yv - pv)))

    base = np.full_like(yv, yv.mean())
    rmse_base = float(np.sqrt(np.mean((yv - base) ** 2)))

    print()
    print("=" * 70)
    print("30 个样品的预测（每个样品都被且只被预测 1 次）")
    print("=" * 70)
    print(f"{'样品':<8}{'实验值':>9}{'预测值':>9}{'误差(mV)':>11}")
    for i, pid in enumerate(ids):
        print(f"{pid:<8}{yv[i]:>9.4f}{pv[i]:>9.4f}{(pv[i] - yv[i]) * 1000:>11.1f}")

    print()
    print("=" * 70)
    print("和论文对比")
    print("=" * 70)
    print(f"  你的 corr = {corr:.4f}      论文 OO-1 = 0.882")
    print(f"  你的 RMSE = {rmse:.4f} V    论文 OO-1 = 0.033 V")
    print(f"  你的 MAE  = {mae:.4f} V    论文 OO-1 = 0.028 V")
    print(f"  [参照] 全部猜平均值的 RMSE = {rmse_base:.4f} V")
    print(f"  你的模型比'什么都不学'好了 {rmse_base - rmse:.4f} V")

    # 与作者预测对照
    res_path = (ROOT / "OER2022" / "initial_train" / "result"
                / "OO_corr=0.882_RMSE=0.033_MAE=0.028_MAPE=5.512_MPE=3.245.csv")
    if res_path.exists():
        res = pd.read_csv(res_path).set_index("id")
        author = res.loc[ids, "pred_avg"].to_numpy(float)
        print(f"  你的预测 vs 作者的预测: corr = {pearson(author, pv):.4f}")
        print(f"  （如果这个数很高，说明你的模型和作者的模型学到了一样的东西）")

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0.4, 0.65], [0.4, 0.65], "k--", lw=1, label="perfect")
        ax.scatter(yv, author, c="#999999", s=30, label="author", alpha=0.7)
        ax.scatter(yv, pv, c="#1f77b4", s=40, label="you")
        ax.set_xlabel("experimental (V)")
        ax.set_ylabel("predicted (V)")
        ax.set_title("Six-fold CV: your model vs author")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "step11_parity.png", dpi=150)
        plt.close(fig)
        print(f"  图已保存: {FIG / 'step11_parity.png'}")

    np.savetxt(
        ROOT / "results" / "step11_predictions.csv",
        np.c_[yv, pv],
        delimiter=",",
        header="experimental,predicted",
        comments="",
    )
    print(f"  预测已存: {ROOT / 'results' / 'step11_predictions.csv'}")


if __name__ == "__main__":
    main()

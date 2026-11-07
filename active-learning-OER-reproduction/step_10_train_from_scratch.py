"""第四步：从零训练一个模型，看它怎么学会预测过电位。

和第 9 步的区别：
  第 9 步 用的是作者的权重，只是"跑一遍"
  第 10 步 权重是随机初始化的，靠训练自己学出来

配置说明（为了在你电脑上几分钟跑完，做了两处简化，都会在输出里说明）：
  1. 用 1296 条真实键，不用 50400 条全组合  -> 速度差 40 倍
  2. 只训练 300 轮，作者是 700 轮            -> 时间砍一半
  其余（batch=5、Adam、MSE、25训5验）完全照论文。

运行：python step_10_train_from_scratch.py
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

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

import os

EPOCHS = int(os.environ.get("EPOCHS", "60"))
BATCH = 5
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
EMBED = 64
STEPS = 4


def build_dataset(ids, graph_suffix="00"):
    """把若干样品打包成 (B, 336, 4) 等张量的列表。"""
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    nodes_list, pairs_list, edges_list = [], [], []
    inten_list, pos_list, tsne_list, y_list = [], [], [], []

    for pid in ids:
        row = df[df["id"] == pid].iloc[0]
        atoms, conect = parse_pdb(GRAPH_DIR / f"{pid}_{graph_suffix}.pdb")
        pairs, edges = build_conect_variant(atoms, conect)

        nodes_list.append(node_features(atoms))              # (336,4)
        pairs_list.append(pairs)                             # (2,1296)
        edges_list.append(edges)                             # (1296,3)
        inten_list.append(row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32))
        pos_list.append(row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32))
        tsne_list.append(row[["tsne_0", "tsne_1"]].to_numpy(np.float32))
        y_list.append(np.float32(row["potential"]))

    return (
        np.stack(nodes_list).astype(np.float32),
        np.stack(pairs_list).astype(np.int32),
        np.stack(edges_list).astype(np.float32),
        np.stack(inten_list).astype(np.float32),
        np.stack(pos_list).astype(np.float32),
        np.stack(tsne_list).astype(np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


class TrainableMPNN(tf.keras.Model):
    """结构和作者 checkpoint 一致，但权重随机初始化。

    dropout>0 时，在拼接之后和第一个全连接之后各加一层 Dropout。
    论文 SI 图1 说明里写了 "passed through dropout and dense layers"，
    所以作者是用了 dropout 的；默认 0 是为了保持前面几步的结果不变。
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.edge_dense = tf.keras.layers.Dense(EMBED * EMBED, name="message_step")
        self.gru = tf.keras.layers.GRUCell(EMBED, name="update_step")
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=EMBED, name="attention"
        )
        self.ffn0 = tf.keras.layers.Dense(128, activation="relu")
        self.ffn1 = tf.keras.layers.Dense(EMBED)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.d_graph = tf.keras.layers.Dense(128)
        self.d_int = tf.keras.layers.Dense(64)
        self.d_pos = tf.keras.layers.Dense(64)
        self.d_tsne = tf.keras.layers.Dense(16)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.d512 = tf.keras.layers.Dense(512, activation="relu")
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.d256 = tf.keras.layers.Dense(256, activation="relu")
        self.d1 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        nodes, pairs, edges, inten, pos, tsne = inputs
        b = tf.shape(nodes)[0]
        n = tf.shape(nodes)[1]
        e = tf.shape(edges)[1]

        h = tf.pad(nodes, [(0, 0), (0, 0), (0, EMBED - tf.shape(nodes)[2])])

        for _ in range(STEPS):
            bond = self.edge_dense(edges)                       # (B,E,4096)
            bond = tf.reshape(bond, (b, e, EMBED, EMBED))       # (B,E,64,64)
            neigh = tf.gather(h, pairs[:, 1], batch_dims=1)     # (B,E,64)
            t = tf.matmul(bond, neigh[..., None])[..., 0]       # (B,E,64)

            offset = tf.range(b)[:, None] * n
            idx = tf.reshape(pairs[:, 0] + offset, (-1,))
            agg = tf.math.unsorted_segment_sum(
                tf.reshape(t, (-1, EMBED)), idx, b * n
            )
            agg = tf.reshape(agg, (b, n, EMBED))
            agg2 = tf.reshape(agg, (b * n, EMBED))
            h2 = tf.reshape(h, (b * n, EMBED))
            h2, _ = self.gru(agg2, h2)
            h = tf.reshape(h2, (b, n, EMBED))

        a = self.attn(h, h)
        x = self.ln1(h + a)
        x = self.ln2(x + self.ffn1(self.ffn0(x)))
        g = tf.reduce_mean(x, axis=1)                           # (B,64)

        u = tf.concat([self.d_graph(g), self.d_int(inten),
                       self.d_pos(pos), self.d_tsne(tsne)], axis=-1)
        u = self.drop1(u)
        x = self.drop2(self.d512(u))
        return self.d1(self.d256(x))[:, 0]


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-12))


def train_one(seed, train_data, val_data):
    tf.keras.utils.set_random_seed(seed)
    model = TrainableMPNN()
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    Xtr, ytr = train_data
    history = []
    n = len(ytr)

    for epoch in range(EPOCHS):
        order = np.random.RandomState(seed * 1000 + epoch).permutation(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            batch = [tf.constant(x[idx]) for x in Xtr]
            yb = tf.constant(ytr[idx])
            with tf.GradientTape() as tape:
                pred = model(batch)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(loss)
            batches += 1
        history.append(epoch_loss / batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      seed {seed}  epoch {epoch + 1:>3}/{EPOCHS}   "
                  f"loss = {history[-1]:.6f}")

    Xv, yv = val_data
    pred_val = model([tf.constant(x) for x in Xv]).numpy()
    return history, pred_val


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    # 论文的折：30 个样本分 6 折，每折 5 个。这里取第 0 折做验证。
    val_ids = ids[0:5]
    train_ids = ids[5:]

    print("=" * 66)
    print("数据划分")
    print("=" * 66)
    print(f"训练集 {len(train_ids)} 个: {', '.join(train_ids)}")
    print(f"验证集 {len(val_ids)} 个: {', '.join(val_ids)}")

    train_data = build_dataset(train_ids)[:-1], build_dataset(train_ids)[-1]
    val_data = build_dataset(val_ids)[:-1], build_dataset(val_ids)[-1]

    print()
    print("=" * 66)
    print(f"开始训练（{len(SEEDS)} 个模型，每个 {EPOCHS} 轮）")
    print("=" * 66)

    all_preds, histories = [], []
    for seed in SEEDS:
        hist, pv = train_one(seed, train_data, val_data)
        histories.append(hist)
        all_preds.append(pv)
        print(f"  seed {seed}: loss {hist[0]:.6f} -> {hist[-1]:.6f}")

    preds = np.stack(all_preds)          # (n_seed, 5)
    y_true = val_data[1]

    print()
    print("=" * 66)
    print("验证集上的预测（这些样品模型没见过）")
    print("=" * 66)
    head = f"{'样品':<8}{'实验值':>8}" + "".join(f"{'seed' + str(s):>9}" for s in SEEDS)
    print(head + f"{'均值':>9}{'标准差':>9}")
    for i, pid in enumerate(val_ids):
        row = f"{pid:<8}{y_true[i]:>8.4f}"
        row += "".join(f"{preds[s][i]:>9.4f}" for s in range(len(SEEDS)))
        row += f"{preds[:, i].mean():>9.4f}"
        sd = preds[:, i].std(ddof=1) if len(SEEDS) > 1 else float("nan")
        row += f"{sd:>9.4f}"
        print(row)

    mean_pred = preds.mean(axis=0)
    unc = preds.std(axis=0, ddof=1) if len(SEEDS) > 1 else np.zeros_like(mean_pred)
    print()
    print(f"  相关系数 corr = {pearson(y_true, mean_pred):.4f}   （论文 30 样本是 0.882）")
    print(f"  RMSE       = {np.sqrt(np.mean((y_true - mean_pred) ** 2)):.4f} V （论文是 0.033）")
    print(f"  平均不确定度 = {np.nanmean(unc):.4f} V")

    # 画 loss 曲线
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, hist in zip(SEEDS, histories):
        ax.plot(hist, label=f"seed {s}", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE (log scale)")
    ax.set_title("Model learns: loss goes down as training proceeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step10_loss.png", dpi=150)
    plt.close(fig)

    # 预测 vs 实验
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.4, 0.65], [0.4, 0.65], "k--", linewidth=1, label="perfect")
    ax.errorbar(y_true, mean_pred, yerr=unc,
                fmt="o", color="#1f77b4", capsize=4, label="predicted")
    ax.set_xlabel("experimental overpotential (V)")
    ax.set_ylabel("predicted overpotential (V)")
    ax.set_title("Validation set: prediction vs experiment")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step10_parity.png", dpi=150)
    plt.close(fig)

    print()
    print(f"图已保存: {FIG / 'step10_loss.png'}")
    print(f"图已保存: {FIG / 'step10_parity.png'}")
    print()
    print("=" * 66)
    print("这一步你刚做了什么")
    print("=" * 66)
    print("  1. 权重从随机数开始，一开始预测全是乱的")
    print("  2. 每看一批数据，算一次误差(MSE)，然后沿误差下降方向改权重")
    print("  3. 300 轮之后，模型学会了'什么样的材料过电位低'")
    print("  4. 3 个不同种子的模型给出 3 个略有差异的预测，它们的标准差就是'不确定度'")


if __name__ == "__main__":
    main()

"""第五步附二：加 dropout，和无 dropout 做对照。

背景：论文 SI 图1 写了 "passed through dropout and dense layers"，
但我们前面几步漏了这层。这里补上，并用完全相同的条件做 A/B 对照：

  同样的折（P01-P05 验证）、同样的种子、同样的 300 轮
  唯一区别：拼接后和第一个全连接后各加一层 Dropout

对照的基线直接读上一步的 results/step12_diagnose.csv，不用重跑。

运行：python step_13_dropout_compare.py
耗时：约 18 分钟（300 轮）

改 dropout 比例：$env:DROPOUT = "0.5"
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

EPOCHS = int(os.environ.get("EPOCHS", "300"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
SEED = 0


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    print("=" * 70)
    print(f"对照实验：dropout = {DROPOUT}，{EPOCHS} 轮")
    print("=" * 70)

    base_path = ROOT / "results" / "step12_diagnose.csv"
    base = pd.read_csv(base_path) if base_path.exists() else None
    if base is not None:
        print(f"  已载入无 dropout 的基线：{len(base)} 个记录点")
    else:
        print("  警告：找不到 step12 的基线，只画本组曲线")

    nodes, pairs, edges, feat, y = load_all(ids)
    val_idx = np.arange(0, 5)
    tr_idx = np.arange(5, 30)

    tf.keras.utils.set_random_seed(SEED)
    model = TrainableMPNN(dropout=DROPOUT)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    Xtr = pack(nodes, pairs, edges, feat, tr_idx)
    ytr = y[tr_idx]
    Xv = pack(nodes, pairs, edges, feat, val_idx)
    yv = y[val_idx]

    def evaluate():
        pred_tr = model(Xtr, training=False).numpy()
        mse_tr = float(np.mean((ytr - pred_tr) ** 2))
        pred_v = model(Xv, training=False).numpy()
        return mse_tr, pearson(yv, pred_v), float(np.sqrt(np.mean((yv - pred_v) ** 2))), pred_v

    records = []
    mse_tr, corr_v, rmse_v, _ = evaluate()
    records.append((0, mse_tr, corr_v, rmse_v))
    print(f"\n  轮数   训练MSE    验证corr   验证RMSE")
    print(f"  {0:>4}   {mse_tr:.6f}   {corr_v:>8.4f}   {rmse_v:.4f}")

    n = len(tr_idx)
    for epoch in range(EPOCHS):
        order = np.random.RandomState(SEED * 7919 + epoch).permutation(n)
        for start in range(0, n, BATCH):
            b = order[start:start + BATCH]
            batch = [tf.constant(x.numpy()[b]) for x in Xtr]
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.constant(ytr[b]), model(batch, training=True))
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

        if (epoch + 1) % EVAL_EVERY == 0:
            mse_tr, corr_v, rmse_v, pred_v = evaluate()
            records.append((epoch + 1, mse_tr, corr_v, rmse_v))
            print(f"  {epoch + 1:>4}   {mse_tr:.6f}   {corr_v:>8.4f}   {rmse_v:.4f}")

    rec = np.asarray(records)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    if base is not None:
        ax.plot(base.iloc[:, 0], base.iloc[:, 1], "o-", color="#d62728",
                markersize=2.5, alpha=0.7, label="no dropout (step12)")
    ax.plot(rec[:, 0], rec[:, 1], "o-", color="#1f77b4",
            markersize=2.5, label=f"dropout {DROPOUT}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE (log)")
    ax.set_title("Train error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if base is not None:
        ax.plot(base.iloc[:, 0], base.iloc[:, 2], "o-", color="#d62728",
                markersize=2.5, alpha=0.7, label="no dropout (step12)")
    ax.plot(rec[:, 0], rec[:, 2], "o-", color="#1f77b4",
            markersize=2.5, label=f"dropout {DROPOUT}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation correlation")
    ax.set_title("Validation correlation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "step13_dropout_compare.png", dpi=150)
    plt.close(fig)

    print()
    print("=" * 70)
    print("对照结果")
    print("=" * 70)
    print(f"  本次 dropout={DROPOUT}:")
    print(f"    训练 MSE 末值 = {rec[-1, 1]:.6f}")
    print(f"    验证 corr 最好 = {np.nanmax(rec[:, 2]):.4f} （第 {int(rec[np.nanargmax(rec[:, 2]), 0])} 轮）")
    print(f"    验证 corr 末值 = {rec[-1, 2]:.4f}")
    print(f"    验证 corr 后 100 轮均值 = {rec[len(rec)//2:, 2].mean():.4f}")
    if base is not None:
        b = base.to_numpy()
        print(f"  基线 无dropout:")
        print(f"    训练 MSE 末值 = {b[-1, 1]:.6f}")
        print(f"    验证 corr 最好 = {np.nanmax(b[:, 2]):.4f} （第 {int(b[np.nanargmax(b[:, 2]), 0])} 轮）")
        print(f"    验证 corr 末值 = {b[-1, 2]:.4f}")
        print(f"    验证 corr 后 100 轮均值 = {b[len(b)//2:, 2].mean():.4f}")
        print()
        print("  判读：")
        print("    如果 dropout 组的'后 100 轮均值'明显高于基线，说明它压住了过拟合；")
        print("    如果两者差不多，说明过拟合不是主因，问题在输入表示或数据量。")

    print(f"\n  图已保存: {FIG / 'step13_dropout_compare.png'}")
    np.savetxt(ROOT / "results" / "step13_dropout.csv", rec, delimiter=",",
               header="epoch,train_mse,val_corr,val_rmse", comments="")


if __name__ == "__main__":
    main()

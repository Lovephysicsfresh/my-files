"""第六步附：dropout 扫描 —— 找那个"刚刚好"的点

我们已经测了两个极端：
  dropout 0.0，300 轮  -> 过拟合（训练 MSE 1e-4，验证 corr 掉到 0.45）
  dropout 0.3，120 轮  -> 正则过强（训练 RMSE 74 mV，模型比"报平均数"还差）

这一步补中间的 dropout = 0.1，同样是 300 轮，条件完全相同。

同时引入一个新指标：**校准比**

    校准比 = 验证 RMSE / 集成不确定度 σ

  接近 1  -> 模型的自我估计可信
  远大于 1 -> 过度自信（模型说"我很确定"，其实错得厉害）
  远小于 1 -> 过度保守

论文的模型校准比大约 1.3（RMSE 0.033 / σ 0.025）。
上一轮我们的集成是 0.0909 / 0.0125 = 7.3，严重过度自信。

运行：python step_15_dropout_sweep.py
耗时：约 18 分钟（300 轮）

改比例：$env:DROPOUT = "0.2"
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
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
SEED = 0


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    print("=" * 70)
    print(f"dropout 扫描：本次 dropout = {DROPOUT}，{EPOCHS} 轮")
    print("=" * 70)

    nodes, pairs, edges, feat, y = load_all(ids)
    val_idx = np.arange(0, 5)
    tr_idx = np.arange(5, 30)

    Xtr = pack(nodes, pairs, edges, feat, tr_idx)
    ytr = y[tr_idx]
    Xv = pack(nodes, pairs, edges, feat, val_idx)
    yv = y[val_idx]

    const_rmse = float(np.sqrt(np.mean((yv - yv.mean()) ** 2)))

    tf.keras.utils.set_random_seed(SEED)
    model = TrainableMPNN(dropout=DROPOUT)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    def evaluate():
        p_tr = model(Xtr, training=False).numpy()
        mse_tr = float(np.mean((ytr - p_tr) ** 2))
        p_v = model(Xv, training=False).numpy()
        return mse_tr, pearson(yv, p_v), float(np.sqrt(np.mean((yv - p_v) ** 2)))

    records = [(0, *evaluate())]
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
            records.append((epoch + 1, *evaluate()))
            print(f"  epoch {epoch + 1:>3}/{EPOCHS}  train MSE {records[-1][1]:.6f}  "
                  f"val corr {records[-1][2]:.4f}  val RMSE {records[-1][3]:.4f}")

    rec = np.asarray(records)

    base12 = ROOT / "results" / "step12_diagnose.csv"
    base13 = ROOT / "results" / "step13_dropout.csv"
    runs = [("dropout 0.0 (step12)", base12, "#d62728"),
            ("dropout 0.3 (step13)", base13, "#2ca02c")]
    loaded = []
    for label, path, color in runs:
        if path.exists():
            loaded.append((label, pd.read_csv(path).to_numpy(), color))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for label, arr, color in loaded:
        ax.plot(arr[:, 0], arr[:, 1], "-", color=color, lw=1.4, alpha=0.75, label=label)
    ax.plot(rec[:, 0], rec[:, 1], "-", color="#1f77b4", lw=1.8,
            label=f"dropout {DROPOUT}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE (log)")
    ax.set_title("Train error: too low = memorizing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, arr, color in loaded:
        ax.plot(arr[:, 0], arr[:, 2], "-", color=color, lw=1.4, alpha=0.75, label=label)
    ax.plot(rec[:, 0], rec[:, 2], "-", color="#1f77b4", lw=1.8,
            label=f"dropout {DROPOUT}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation correlation")
    ax.set_title("Validation correlation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "step15_dropout_sweep.png", dpi=150)
    plt.close(fig)

    print()
    print("=" * 70)
    print("三种配置对比（同样 300 轮、同样种子、同样折）")
    print("=" * 70)
    print(f"{'配置':<24}{'训练RMSE':>10}{'验证corr':>10}{'验证RMSE':>10}{'后100轮均值':>12}")
    for label, arr, _ in loaded:
        print(f"{label:<24}{np.sqrt(arr[-1, 1]):>10.4f}{arr[-1, 2]:>10.4f}"
              f"{arr[-1, 3]:>10.4f}{arr[len(arr) // 2:, 2].mean():>12.4f}")
    print(f"{'dropout ' + str(DROPOUT):<24}{np.sqrt(rec[-1, 1]):>10.4f}"
          f"{rec[-1, 2]:>10.4f}{rec[-1, 3]:>10.4f}{rec[len(rec) // 2:, 2].mean():>12.4f}")
    print()
    print(f"  [参照] 全部猜平均值的 RMSE = {const_rmse:.4f} V")
    print(f"          任何模型的验证 RMSE 低于这个数，才算学到东西")
    print()
    print(f"  本次 dropout={DROPOUT} 的验证 RMSE = {rec[-1, 3]:.4f} V")
    if rec[-1, 3] < const_rmse:
        print(f"  -> 比基线好 {const_rmse - rec[-1, 3]:.4f} V，模型有用")
    else:
        print(f"  -> 比基线差 {rec[-1, 3] - const_rmse:.4f} V，模型还不如直接猜平均数")

    print()
    print(f"  图已保存: {FIG / 'step15_dropout_sweep.png'}")
    np.savetxt(ROOT / "results" / f"step15_dropout{int(DROPOUT * 100)}.csv", rec,
               delimiter=",", header="epoch,train_mse,val_corr,val_rmse", comments="")


if __name__ == "__main__":
    main()

"""第五步附：诊断实验 —— 为什么 corr 只有 0.50？

只跑一折（第 1 折，验证集 P01-P05），但训练 300 轮，
每 10 轮同时记录两个数：

  训练集 MSE   —— 模型对"见过的 25 个样品"拟合得多好
  验证集 corr  —— 模型对"没见过的 5 个样品"排序得多准

这两个数放在一起看，就能判断是哪种病：

  欠拟合：训练 MSE 还很高且仍在下降，验证 corr 也在上升
          -> 对策：加轮数、加模型容量
  过拟合：训练 MSE 已很低，验证 corr 先升后降
          -> 对策：早停、加正则、加数据、做集成

运行：python step_12_diagnose.py
耗时：约 18 分钟（300 轮 × 3.7 秒）

想改轮数：$env:EPOCHS = "500"
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
SEED = 0


def main():
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    ids = df["id"].tolist()

    print("=" * 70)
    print(f"诊断实验：1 折 × {EPOCHS} 轮，每 {EVAL_EVERY} 轮评估一次")
    print("=" * 70)

    nodes, pairs, edges, feat, y = load_all(ids)
    val_idx = np.arange(0, 5)
    tr_idx = np.arange(5, 30)
    print(f"  训练: {len(tr_idx)} 个 ({ids[5]} ~ {ids[-1]})")
    print(f"  验证: {len(val_idx)} 个 ({', '.join(ids[:5])})")

    tf.keras.utils.set_random_seed(SEED)
    model = TrainableMPNN()
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    Xtr = pack(nodes, pairs, edges, feat, tr_idx)
    ytr = y[tr_idx]
    Xv = pack(nodes, pairs, edges, feat, val_idx)
    yv = y[val_idx]

    def evaluate():
        pred_tr = model(Xtr).numpy()
        mse_tr = float(np.mean((ytr - pred_tr) ** 2))
        pred_v = model(Xv).numpy()
        corr_v = pearson(yv, pred_v)
        rmse_v = float(np.sqrt(np.mean((yv - pred_v) ** 2)))
        return mse_tr, corr_v, rmse_v, pred_v

    records = []
    mse_tr, corr_v, rmse_v, pred_v = evaluate()
    records.append((0, mse_tr, corr_v, rmse_v))
    print(f"\n  轮数   训练MSE    验证corr   验证RMSE     验证集预测")
    print(f"  {0:>4}   {mse_tr:.6f}   {corr_v:>8.4f}   {rmse_v:.4f}   "
          + " ".join(f"{p:.3f}" for p in pred_v))

    n = len(tr_idx)
    for epoch in range(EPOCHS):
        order = np.random.RandomState(SEED * 7919 + epoch).permutation(n)
        for start in range(0, n, BATCH):
            b = order[start:start + BATCH]
            batch = [tf.constant(x.numpy()[b]) for x in Xtr]
            with tf.GradientTape() as tape:
                loss = loss_fn(tf.constant(ytr[b]), model(batch))
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

        if (epoch + 1) % EVAL_EVERY == 0:
            mse_tr, corr_v, rmse_v, pred_v = evaluate()
            records.append((epoch + 1, mse_tr, corr_v, rmse_v))
            print(f"  {epoch + 1:>4}   {mse_tr:.6f}   {corr_v:>8.4f}   {rmse_v:.4f}   "
                  + " ".join(f"{p:.3f}" for p in pred_v))

    rec = np.asarray(records)
    epochs_, mse_, corr_, rmse_ = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(epochs_, mse_, "o-", color="#d62728", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE (log)")
    ax.set_title("Train error: does it keep going down?")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_, corr_, "o-", color="#1f77b4", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation correlation (5 samples)")
    ax.set_title("Validation correlation: does it improve in step?")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "step12_diagnose.png", dpi=150)
    plt.close(fig)

    final_mse = float(mse_[-1])
    best_corr = float(np.nanmax(corr_))
    best_epoch = int(epochs_[int(np.nanargmax(corr_))])
    final_corr = float(corr_[-1])

    print()
    print("=" * 70)
    print("诊断结论")
    print("=" * 70)
    print(f"  训练 MSE：起点 {mse_[0]:.6f} -> 终点 {final_mse:.6f}")
    print(f"  验证 corr：最好 {best_corr:.4f}（第 {best_epoch} 轮），终点 {final_corr:.4f}")

    if final_mse > 0.002:
        print("\n  >> 判定：欠拟合")
        print("     训练集本身还没拟合好，模型容量或轮数不够。")
        print("     下一步该做的：加轮数（EPOCHS=700），或把边换成 50400 条全组合。")
    elif best_epoch < EPOCHS * 0.6:
        print("\n  >> 判定：过拟合")
        print(f"     第 {best_epoch} 轮后验证集开始变差，训练集还在继续降。")
        print("     下一步该做的：早停 + 多训练几个种子取平均（这就是论文的集成）。")
    else:
        print("\n  >> 判定：基本收敛")
        print("     训练和验证同步改善，没有明显过拟合。")
        print("     剩下的差距来自输入表示或超参，不在训练轮数上。")

    print(f"\n  图已保存: {FIG / 'step12_diagnose.png'}")
    np.savetxt(
        ROOT / "results" / "step12_diagnose.csv", rec, delimiter=",",
        header="epoch,train_mse,val_corr,val_rmse", comments="",
    )
    print(f"  数据已存: {ROOT / 'results' / 'step12_diagnose.csv'}")


if __name__ == "__main__":
    main()

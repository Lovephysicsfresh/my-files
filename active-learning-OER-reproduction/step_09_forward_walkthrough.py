"""第三步：跟着一个材料走一遍神经网络，把每一步的中间结果打印出来。

目的：让你亲眼看到 336 个原子 -> ... -> 一个预测值 0.546 的过程。

必须用 run.bat 跑（需要 TensorFlow）：
    .\run.bat step_09_forward_walkthrough.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import (
    GRAPH_DIR,
    build_dense_variant,
    node_features,
    parse_pdb,
)
from step_04_load_checkpoint import Net

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "OER2022" / "initial_train" / "checkpoint" / "OO_00_0"


def line(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def peek(arr, n=6):
    a = np.asarray(arr).ravel()
    txt = ", ".join(f"{v:.4f}" for v in a[:n])
    return f"[{txt}, ...]"


def main():
    line("第 0 步：准备输入")
    df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    row = df[df["id"] == "P01"].iloc[0]
    intensity = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
    position = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
    tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)
    truth = row["potential"]

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    nodes = node_features(atoms)
    pairs, edges = build_dense_variant(atoms, conect)
    pairs = tf.constant(pairs, tf.int32)
    edges = tf.constant(edges, tf.float32)

    print(f"原子特征矩阵 nodes      : {nodes.shape}")
    print(f"边索引 pairs            : {tuple(pairs.shape)}  (2 行 = 起点,终点)")
    print(f"边特征 edges            : {tuple(edges.shape)}  (3 列 = A-O, B-O, A-B)")
    print(f"XRD 强度 intensity      : {intensity.shape}  {peek(intensity)}")
    print(f"XRD 峰位 position       : {position.shape}  {peek(position)}")
    print(f"成分 t-SNE (2维)        : {tsne.shape}  {peek(tsne)}")
    print(f"真实答案 potential      : {truth} V")

    net = Net(CKPT).load()

    # ---------------- 1. 补零 ----------------
    line("第 1 步：把每个原子的 4 个数补成 64 个数")
    h = tf.constant(nodes, tf.float32)
    h = tf.pad(h, [(0, 0), (0, 64 - tf.shape(h)[1])])
    print(f"补零前: (336, 4)")
    print(f"补零后: {tuple(h.shape)}")
    print(f"第 1 个原子: {peek(h[0], 8)}")
    print("说明：后面 60 个位置全是 0，只是为了让维度对上 64。")

    # ---------------- 2. 消息传递 ----------------
    line("第 2 步：消息传递（重复 4 次）")
    for step in range(4):
        bond = net.edge_dense(edges)                       # (E, 4096)
        bond3 = tf.reshape(bond, (-1, 64, 64))             # (E, 64, 64)
        neigh = tf.gather(h, pairs[1])                     # (E, 64)
        t = tf.matmul(bond3, neigh[..., None])[..., 0]     # (E, 64)
        agg = tf.math.unsorted_segment_sum(
            t, pairs[0], num_segments=tf.shape(h)[0]
        )                                                  # (336, 64)
        h, _ = net.gru_cell(agg, h)

        if step == 0:
            print(f"  边特征喂进 Dense 后     : {tuple(bond.shape)}")
            print(f"  重塑成每条边一个矩阵    : {tuple(bond3.shape)}")
            print(f"  这条边两端是原子 {int(pairs[0][0])} 和 {int(pairs[1][0])}")
            print(f"  边权重矩阵左上角 3x3    : {peek(bond3[0], 6)}")
            print(f"  邻居原子的向量          : {peek(neigh[0], 6)}")
            print(f"  两者相乘得到消息        : {peek(t[0], 6)}")
        print(f"  第 {step + 1} 轮结束，节点状态: {tuple(h.shape)}  首个节点 {peek(h[0])}")
    print("说明：每一轮，每个原子都从邻居那里收到一条'消息'，再用 GRU 更新自己。")
    print("      4 轮之后，一个原子间接知道 4 跳以内的所有邻居。")

    # ---------------- 3. 注意力 ----------------
    line("第 3 步：注意力 + 池化，把 336 个原子压成 1 个总向量")
    attn = net.attention(h[None], h[None], h[None])[0]
    x = net.ln1(h + attn)
    x = net.ln2(x + net.ffn_1(net.ffn_0(x)))
    g = tf.reduce_mean(x, axis=0)
    print(f"注意力输出              : {tuple(attn.shape)}")
    print(f"每个原子的最终状态      : {tuple(x.shape)}")
    print(f"对 336 个原子取平均     : {tuple(g.shape)}")
    print(f"图的总向量              : {peek(g)}")
    print("说明：注意力让模型自己决定'哪些原子更重要'。取平均则把整个晶体压成一个向量。")

    # ---------------- 4. 四路分支 ----------------
    line("第 4 步：四路特征各自过一层 Dense")
    g128 = net.d_graph(g[None])[0]
    i64 = net.d_int(tf.constant(intensity[None]))[0]
    p64 = net.d_pos(tf.constant(position[None]))[0]
    t16 = net.d_tsne(tf.constant(tsne[None]))[0]
    print(f"图向量       64 -> 128  : {tuple(g128.shape)}  {peek(g128)}")
    print(f"XRD 强度      7 ->  64  : {tuple(i64.shape)}  {peek(i64)}")
    print(f"XRD 峰位      7 ->  64  : {tuple(p64.shape)}  {peek(p64)}")
    print(f"成分 t-SNE    2 ->  16  : {tuple(t16.shape)}  {peek(t16)}")

    # ---------------- 5. 拼接 ----------------
    line("第 5 步：拼接")
    u = tf.concat([g128, i64, p64, t16], axis=0)
    print(f"128 + 64 + 64 + 16 = {u.shape[0]}")
    print(f"拼接结果: {tuple(u.shape)}")
    print(f"前 6 个数: {peek(u)}")

    # ---------------- 6. 回归头 ----------------
    line("第 6 步：三层全连接，吐出最终数字")
    h1 = net.d512(u[None])[0]
    h2 = net.d256(h1[None])[0]
    out = float(net.d1(h2[None])[0, 0])
    print(f"272 -> 512 : {tuple(h1.shape)}  {peek(h1)}")
    print(f"512 -> 256 : {tuple(h2.shape)}  {peek(h2)}")
    print(f"256 ->   1 : 标量")
    print()
    print(f"  模型预测的过电位 = {out:.4f} V")
    print(f"  实验测得的过电位 = {truth:.4f} V")
    print(f"  误差 = {abs(out - truth) * 1000:.0f} mV")

    line("总结")
    print("  输入:  336 个原子 x 4 个数 + 50400 条边 + 14 个 XRD 数 + 2 个成分坐标")
    print("  输出:  1 个数（过电位，单位 V）")
    print("  中间:  补零 -> 4 轮消息传递 -> 注意力 -> 平均 -> 四路 Dense -> 拼接 -> 三层全连接")
    print()
    print("  作者用 50 个这样的模型各跑一次，得到 10 个预测值，")
    print("  取平均当预测结果，取标准差当'不确定度'。这一步就是论文里的 bootstrap。")


if __name__ == "__main__":
    main()

"""第7步：检验"自环"假设。

Keras 官方 MPNN 示例里，每个节点都会追加一条自环 [i, i]，
边特征用"无键"槽位 [0,0,1] 占位。

本论文严格照抄了该示例的结构（从 checkpoint 可证），因此很可能保留了自环。
如果保留，那 3 个 bond 特征的第 3 列就不是 A-B 键，而是"自环/无键"。

三种写法对比：
  A) 无自环，A-B 用第3列
  B) 加自环，第3列为 A-B，自环三列全 0
  C) 加自环，第3列为"自环/无键"
"""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import parse_pdb, GRAPH_DIR, ELECTRONEGATIVITY, IONIC_RADIUS
from step_04_load_checkpoint import Net

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "OER2022" / "initial_train" / "checkpoint"
TARGET = 0.534791


def nodes_of(atoms):
    return np.asarray([[
        IONIC_RADIUS[a["element"]],
        ELECTRONEGATIVITY[a["element"]],
        1.0 if a["site"] == "A" else 0.0,
        1.0 if a["site"] == "B" else 0.0,
    ] for a in atoms], np.float32)


def build(atoms, conect, self_loop, third_slot):
    index_of = {a["serial"]: i for i, a in enumerate(atoms)}
    connected = {tuple(sorted((index_of[u], index_of[v]))) for u, v in conect}
    by = {"A": [], "B": [], "O": []}
    for i, a in enumerate(atoms):
        by[a["site"]].append(i)

    pairs, edges = [], []
    for s1, s2 in (("A", "O"), ("B", "O"), ("A", "B")):
        if third_slot == "AB" and s1 == "A" and s2 == "B":
            vec = [0.0, 0.0, 1.0]
        elif s1 == "A":
            vec = [1.0, 0.0, 0.0]
        else:
            vec = [0.0, 1.0, 0.0]
        for i, j in product(by[s1], by[s2]):
            f = 1.0 if tuple(sorted((i, j))) in connected else 0.0
            for u, v in ((i, j), (j, i)):
                pairs.append([u, v])
                edges.append([c * f for c in vec])

    if self_loop:
        for i in range(len(atoms)):
            pairs.append([i, i])
            edges.append([0.0, 0.0, 1.0])

    return (tf.constant(np.asarray(pairs, np.int32).T, tf.int32),
            tf.constant(np.asarray(edges, np.float32), tf.float32))


def main():
    df = pd.read_csv(PROJECT_ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    row = df[df["id"] == "P01"].iloc[0]
    inten = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
    pos = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
    tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    nodes = nodes_of(atoms)
    net = Net(CKPT_DIR / "OO_00_0").load()

    print(f"目标 {TARGET}\n")
    for tag, sl, third in [
        ("A 无自环, 第3列=A-B", False, "AB"),
        ("B 有自环, 第3列=A-B", True, "AB"),
        ("C 有自环, 第3列=自环占位", True, "none"),
        ("D 无自环, 第3列=自环占位", False, "none"),
    ]:
        p, e = build(atoms, conect, sl, third)
        pred = net.predict(nodes, p, e, inten, pos, tsne)
        print(f"{tag:26s} E={e.shape[0]:6d}  pred={pred:.6f}  Δ={abs(pred - TARGET):.4f}")


if __name__ == "__main__":
    main()

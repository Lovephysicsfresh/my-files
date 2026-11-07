"""第6步：闭合最后 0.0115 V —— 枚举离子半径口径。

论文只写 "ionic radius"，没给表。但它在第2节引用了 Bartel 等人的新容忍因子，
而容忍因子算的是 A/B 位**按氧化态取**的 Shannon 半径。所以最可能的做法是：
由电荷中性推出 B 位氧化态，再查对应价态的半径。

ABO3 电荷中性：
    sum_A x_i * v_i + sum_B x_j * v_j = 6      (O 为 -2，三个 O)
A 位取常见价（La3+, Ba2+, Sr2+, Pr3+, Ca2+），则
    B 平均价 = 6 - A 平均价

本脚本比较 5 种口径，看哪种让 fold0 命中 0.534791。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import (
    ELECTRONEGATIVITY,
    GRAPH_DIR,
    build_dense_variant,
    parse_pdb,
)
from step_04_load_checkpoint import Net

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "OER2022" / "initial_train" / "checkpoint"
TARGET = 0.534791

A_VALENCE = {"La": 3, "Ba": 2, "Sr": 2, "Pr": 3, "Ca": 2}
A_RADIUS = {"La": 1.36, "Ba": 1.61, "Sr": 1.44, "Pr": 1.30, "Ca": 1.34}

# 六配位 Shannon 半径，按价态
B_RADIUS = {
    "Mn": {2: 0.83, 3: 0.645, 4: 0.53},
    "Fe": {2: 0.78, 3: 0.645, 4: 0.585},
    "Co": {2: 0.745, 3: 0.545, 4: 0.53},
    "Ni": {2: 0.69, 3: 0.56, 4: 0.48},
    "Cu": {2: 0.73, 3: 0.54},
}

# 方案1：固定高价态（之前用的）
FIXED = {"Mn": 0.645, "Fe": 0.645, "Co": 0.545, "Ni": 0.560, "Cu": 0.730}


def b_valence_from_composition(row) -> int:
    a_avg = sum(row[f"atom_{el}"] * v for el, v in A_VALENCE.items())
    b_avg = 6.0 - a_avg
    v = int(round(b_avg))
    return min(max(v, 2), 4)


def make_nodes(atoms, radius_b_lookup, scale_radius=1.0, standardize=False):
    rows = []
    for a in atoms:
        el = a["element"]
        if a["site"] == "A":
            r = A_RADIUS[el]
        elif a["site"] == "B":
            r = radius_b_lookup[el]
        else:
            r = 1.40
        rows.append([
            r * scale_radius,
            ELECTRONEGATIVITY[el],
            1.0 if a["site"] == "A" else 0.0,
            1.0 if a["site"] == "B" else 0.0,
        ])
    arr = np.asarray(rows, np.float32)
    if standardize:
        arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
    return arr


def main():
    df = pd.read_csv(PROJECT_ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    row = df[df["id"] == "P01"].iloc[0]
    intensity = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
    position = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
    tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    pairs, edges = build_dense_variant(atoms, conect)
    pairs_tf = tf.constant(pairs, tf.int32)
    edges_tf = tf.constant(edges, tf.float32)

    net = Net(CKPT_DIR / "OO_00_0").load()
    print(f"目标 fold0 = {TARGET}\n")

    v = b_valence_from_composition(row)
    print(f"P01 = LaMnO3，按电荷中性 B 位价态 = +{v}\n")

    configs = []

    # 1 固定高价态
    configs.append(("固定高价态", lambda a: make_nodes(a, FIXED)))

    # 2 按电荷中性价态
    dyn = {}
    for el in ("Mn", "Fe", "Co", "Ni", "Cu"):
        table = B_RADIUS[el]
        dyn[el] = table[v] if v in table else table[max(table)]
    configs.append((f"电荷中性价态 +{v}", lambda a: make_nodes(a, dyn)))

    # 3 半径用 pm
    configs.append(("半径用pm(x100)", lambda a: make_nodes(a, dyn, scale_radius=100.0)))

    # 4 标准化
    configs.append(("4特征z-score", lambda a: make_nodes(a, dyn, standardize=True)))

    # 5 半径置0，只用其他3维
    zero_r = {el: 0.0 for el in dyn}
    configs.append(("半径置0", lambda a: make_nodes(a, zero_r)))

    best = None
    for name, fn in configs:
        nodes = fn(atoms)
        pred = net.predict(nodes, pairs_tf, edges_tf, intensity, position, tsne)
        d = abs(pred - TARGET)
        flag = ""
        if best is None or d < best[1]:
            best = (name, d)
            flag = "  <-- 当前最优"
        print(f"{name:22s} pred={pred:.6f}  Δ={d:.4f}{flag}")

    print(f"\n最优：{best[0]}，Δ={best[1]:.4f}")


if __name__ == "__main__":
    main()

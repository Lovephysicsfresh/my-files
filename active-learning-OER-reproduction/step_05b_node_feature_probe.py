"""第5b步：节点特征处置方式的探索。

SI 原话是"Each metal atom was represented as a vector of four features"，
说明**氧原子可能不是 4 维特征，而是全 0**。这里枚举 4 种组合，
看哪种能让 fold0 命中作者值 0.534791。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from step_03_build_graph import (
    ELECTRONEGATIVITY,
    GRAPH_DIR,
    IONIC_RADIUS,
    build_conect_variant,
    build_dense_variant,
    parse_pdb,
)
from step_04_load_checkpoint import Net

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "OER2022" / "initial_train" / "checkpoint"
TARGET = 0.534791


def nodes_with_mode(atoms, zero_oxygen: bool):
    out = []
    for a in atoms:
        if zero_oxygen and a["site"] == "O":
            out.append([0.0, 0.0, 0.0, 0.0])
        else:
            out.append([
                IONIC_RADIUS[a["element"]],
                ELECTRONEGATIVITY[a["element"]],
                1.0 if a["site"] == "A" else 0.0,
                1.0 if a["site"] == "B" else 0.0,
            ])
    return np.asarray(out, np.float32)


def main():
    df = pd.read_csv(PROJECT_ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    row = df[df["id"] == "P01"].iloc[0]
    intensity = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
    position = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
    tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    variants = {
        "A_1296": build_conect_variant(atoms, conect),
        "B_50400": build_dense_variant(atoms, conect),
    }

    net = Net(CKPT_DIR / "OO_00_0").load()
    print(f"目标 fold0 = {TARGET}\n")

    for zero_o in (False, True):
        nodes = nodes_with_mode(atoms, zero_o)
        for name, (pairs, edges) in variants.items():
            pred = net.predict(
                nodes,
                tf.constant(pairs, tf.int32),
                tf.constant(edges, tf.float32),
                intensity, position, tsne,
            )
            print(
                f"氧特征={'全0' if zero_o else '正常'}  {name}: "
                f"pred={pred:.6f}  Δ={abs(pred - TARGET):.4f}"
            )


if __name__ == "__main__":
    main()

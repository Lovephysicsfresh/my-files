"""第5步：用 seed-00 的 5 个折模型判定"边怎么建"。

原理：
作者 result CSV 里，P01 的 `0` 列 = 0.534791，这是 seed 0 下某个折模型
（把 P01 放进验证集的那个折）给出的预测。5 个折模型 OO_00_0..OO_00_4 里
必然有一个能复现 0.534791。

于是：把两种边构造方案各跑 5 个折模型，谁的输出能对上 0.534791，谁就是对的。
"""

from pathlib import Path

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
from step_04_load_checkpoint import Net

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "OER2022" / "initial_train" / "checkpoint"
TARGET = 0.534791  # 作者 P01 的 seed-0 预测


def main():
    df = pd.read_csv(PROJECT_ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    row = df[df["id"] == "P01"].iloc[0]
    intensity = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
    position = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
    tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    nodes = node_features(atoms)

    variants = {
        "A_1296": build_conect_variant(atoms, conect),
        "B_50400": build_dense_variant(atoms, conect),
    }

    print(f"目标值（作者 P01, seed0）= {TARGET}\n")

    for fold in range(5):
        ckpt = CKPT_DIR / f"OO_00_{fold}"
        if not (Path(str(ckpt) + ".index")).exists():
            print(f"fold {fold}: checkpoint 缺失，跳过")
            continue
        net = Net(ckpt).load()
        line = [f"fold {fold}:"]
        for name, (pairs, edges) in variants.items():
            pred = net.predict(
                nodes,
                tf.constant(pairs, tf.int32),
                tf.constant(edges, tf.float32),
                intensity, position, tsne,
            )
            line.append(f"{name}={pred:.6f} (Δ={abs(pred - TARGET):.4f})")
        print("  ".join(line))


if __name__ == "__main__":
    main()

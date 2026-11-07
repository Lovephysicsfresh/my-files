"""第3步：把 PDB 图转成模型输入张量。

产出（每个 PDB 一份）：
    node_features : (336, 4)   [离子半径, Pauling电负性, A位标识, B位标识]
    edge_features : (N_e, 3)   [A-O, B-O, A-B] 连接状态 0/1
    pair_indices  : (2, N_e)   [源节点, 目标节点]

两种 pair 集合都生成，供后面用 checkpoint 做判别实验：
    variant="conect"  : 只取 PDB CONECT 里真实存在的键（1296 条）
    variant="dense"   : 取所有 A-O / B-O / A-B 组合（25200 条），未连接记 0

注意：离子半径取的是 Shannon 六配位/十二配位常用值，氧化态与自旋态的选择
在论文中没有逐条列出，属于"待核对假设"，见 复现手册.md 第 3 节。
"""

from pathlib import Path
from itertools import product

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
GRAPH_DIR = PROJECT_ROOT / "OER2022" / "initial_train" / "graph"

A_ELEMENTS = ["La", "Ba", "Sr", "Pr", "Ca"]
B_ELEMENTS = ["Mn", "Fe", "Co", "Ni", "Cu"]

# Pauling 电负性
ELECTRONEGATIVITY = {
    "La": 1.10, "Ba": 0.89, "Sr": 0.95, "Pr": 1.13, "Ca": 1.00,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90,
    "O": 3.44,
}

# Shannon 离子半径（Å）。A位取12配位，B位取6配位高价态。
IONIC_RADIUS = {
    "La": 1.36, "Ba": 1.61, "Sr": 1.44, "Pr": 1.30, "Ca": 1.34,
    "Mn": 0.645, "Fe": 0.645, "Co": 0.545, "Ni": 0.560, "Cu": 0.730,
    "O": 1.40,
}


def site_of(element: str) -> str:
    if element in A_ELEMENTS:
        return "A"
    if element in B_ELEMENTS:
        return "B"
    if element == "O":
        return "O"
    raise ValueError(f"未知元素: {element}")


def parse_pdb(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()

    atoms = []
    conect = []

    for line in lines:
        if line.startswith("HETATM"):
            element = line[76:78].strip()
            atoms.append({
                "serial": int(line[6:11]),
                "element": element,
                "site": site_of(element),
                "xyz": np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]),
            })
        elif line.startswith("CONECT"):
            values = [
                int(line[i:i + 5])
                for i in range(6, len(line), 5)
                if line[i:i + 5].strip()
            ]
            conect.extend((values[0], target) for target in values[1:])

    return atoms, conect


def node_features(atoms) -> np.ndarray:
    features = []
    for atom in atoms:
        element = atom["element"]
        features.append([
            IONIC_RADIUS[element],
            ELECTRONEGATIVITY[element],
            1.0 if atom["site"] == "A" else 0.0,
            1.0 if atom["site"] == "B" else 0.0,
        ])
    return np.asarray(features, dtype=np.float32)


def bond_vector(site_u: str, site_v: str) -> list:
    pair = {site_u, site_v}
    if pair == {"A", "O"}:
        return [1.0, 0.0, 0.0]
    if pair == {"B", "O"}:
        return [0.0, 1.0, 0.0]
    if pair == {"A", "B"}:
        return [0.0, 0.0, 1.0]
    return [0.0, 0.0, 0.0]


def build_conect_variant(atoms, conect):
    index_of = {atom["serial"]: i for i, atom in enumerate(atoms)}
    pairs, edges = [], []
    for u, v in conect:
        iu, iv = index_of[u], index_of[v]
        pairs.append([iu, iv])
        edges.append(bond_vector(atoms[iu]["site"], atoms[iv]["site"]))
    return np.asarray(pairs, dtype=np.int64).T, np.asarray(edges, dtype=np.float32)


def build_dense_variant(atoms, conect):
    index_of = {atom["serial"]: i for i, atom in enumerate(atoms)}
    connected = {tuple(sorted((index_of[u], index_of[v]))) for u, v in conect}

    by_site = {"A": [], "B": [], "O": []}
    for i, atom in enumerate(atoms):
        by_site[atom["site"]].append(i)

    pairs, edges = [], []
    for s1, s2 in (("A", "O"), ("B", "O"), ("A", "B")):
        vec = bond_vector(s1, s2)
        for i, j in product(by_site[s1], by_site[s2]):
            key = tuple(sorted((i, j)))
            flag = 1.0 if key in connected else 0.0
            pairs.append([i, j])
            pairs.append([j, i])
            edges.append([c * flag for c in vec])
            edges.append([c * flag for c in vec])

    return np.asarray(pairs, dtype=np.int64).T, np.asarray(edges, dtype=np.float32)


if __name__ == "__main__":
    sample = GRAPH_DIR / "P01_00.pdb"
    atoms, conect = parse_pdb(sample)

    x = node_features(atoms)
    print("node_features", x.shape)
    assert x.shape == (336, 4)

    p_conect, e_conect = build_conect_variant(atoms, conect)
    print("conect variant  pair_indices", p_conect.shape, "edge_features", e_conect.shape)
    assert p_conect.shape == (2, 1296)
    assert e_conect.shape == (1296, 3)

    p_dense, e_dense = build_dense_variant(atoms, conect)
    print("dense variant   pair_indices", p_dense.shape, "edge_features", e_dense.shape)
    print("dense 连接为1的边数", int(e_dense.sum()), "（两方向合计）")

    print("A-O 边数", int(e_conect[:, 0].sum()))
    print("B-O 边数", int(e_conect[:, 1].sum()))
    print("A-B 边数", int(e_conect[:, 2].sum()))

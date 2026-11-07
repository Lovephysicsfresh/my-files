from pathlib import Path
from collections import Counter
import json

ROOT = Path(__file__).resolve().parent / "OER2022"
A_ELEMENTS = {"La", "Ba", "Sr", "Pr", "Ca"}
B_ELEMENTS = {"Mn", "Fe", "Co", "Ni", "Cu"}


def parse_graph(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms = []
    for line in lines:
        if line.startswith("HETATM"):
            label = line[12:16].strip()
            element = line[76:78].strip()
            if element in A_ELEMENTS:
                site = "A"
            elif element in B_ELEMENTS:
                site = "B"
            elif element == "O":
                site = "O"
            else:
                site = "?"
            atoms.append({
                "serial": int(line[6:11]),
                "label": label,
                "element": element,
                "site": site,
                "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            })
    edges = []
    for line in lines:
        if line.startswith("CONECT"):
            vals = [int(line[i:i + 5]) for i in range(6, len(line), 5) if line[i:i + 5].strip()]
            edges.extend((vals[0], v) for v in vals[1:])
    return atoms, edges


def bond_type(site_u, site_v):
    pair = tuple(sorted((site_u, site_v)))
    if pair == ("A", "O"):
        return "A-O"
    if pair == ("B", "O"):
        return "B-O"
    if pair == ("A", "B"):
        return "A-B"
    return "-".join(pair)


if __name__ == "__main__":
    for name in ["P01_00.pdb", "P07_00.pdb", "P20_00.pdb", "P30_00.pdb"]:
        atoms, edges = parse_graph(ROOT / "initial_train" / "graph" / name)
        serial_to_site = {a["serial"]: a["site"] for a in atoms}
        counts = Counter(bond_type(serial_to_site[u], serial_to_site[v]) for u, v in edges)
        print(name, "nodes", len(atoms), "edges", len(edges), dict(counts))
        print("   elements", dict(Counter(a["element"] for a in atoms)))

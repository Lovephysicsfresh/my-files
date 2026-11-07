from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
PDB_PATH = PROJECT_ROOT / "OER2022" / "initial_train" / "perovskite_oxide.pdb"
OUT = PROJECT_ROOT / "results" / "step_02_pdb_graph.json"

lines = PDB_PATH.read_text(encoding="utf-8").splitlines()
atoms = []
edges = []
for line in lines:
    if line.startswith("HETATM"):
        label = line[12:16].strip()
        site = "A" if label.startswith("AA") else "B" if label.startswith("BB") else "O" if label.startswith("O") else "unknown"
        atoms.append({
            "serial": int(line[6:11]),
            "label": label,
            "element": line[76:78].strip(),
            "site": site,
            "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
        })
    elif line.startswith("CONECT"):
        vals = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
        edges.extend((vals[0], target) for target in vals[1:])

serial_to_idx = {a["serial"]: i for i, a in enumerate(atoms)}
pairs = [[serial_to_idx[u], serial_to_idx[v]] for u, v in edges]
assert len(atoms) == 336
assert len(edges) == 1296
assert len({tuple(sorted(e)) for e in pairs}) == 648
assert all(0 <= i < len(atoms) for e in pairs for i in e)

site_counts = {s: sum(a["site"] == s for a in atoms) for s in ("A", "B", "O")}
report = {
    "n_nodes": len(atoms),
    "n_directed_edges": len(pairs),
    "n_unique_undirected_edges": len({tuple(sorted(e)) for e in pairs}),
    "pair_indices_shape": [2, len(pairs)],
    "site_counts": site_counts,
    "node_feature_shape_placeholder": [len(atoms), 4],
    "edge_feature_shape_placeholder": [len(pairs), 3],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

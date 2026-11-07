from pathlib import Path
from collections import Counter
import json

PDB = Path(__file__).resolve().parent / "OER2022" / "initial_train" / "perovskite_oxide.pdb"
OUT = Path(__file__).resolve().parent / "results" / "pdb_audit.json"
OUT.parent.mkdir(exist_ok=True)

lines = PDB.read_text(encoding="utf-8").splitlines()
atoms = []
edges = []
for line in lines:
    if line.startswith("HETATM"):
        atoms.append({
            "serial": int(line[6:11]),
            "label": line[12:16].strip(),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": line[76:78].strip(),
        })
    elif line.startswith("CONECT"):
        values = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
        if values:
            source = values[0]
            edges.extend((source, target) for target in values[1:])

assert len(atoms) == 336, len(atoms)
assert len(edges) == 1296, len(edges)
assert len({tuple(sorted(e)) for e in edges}) == 648
assert sorted(a["serial"] for a in atoms) == list(range(1, 337))

report = {
    "pdb": str(PDB),
    "atoms": len(atoms),
    "directed_edges": len(edges),
    "unique_undirected_edges": len({tuple(sorted(e)) for e in edges}),
    "label_counts": dict(Counter(a["label"] for a in atoms)),
    "element_counts": dict(Counter(a["element"] for a in atoms)),
    "coordinate_min": [min(a[k] for a in atoms) for k in ("x", "y", "z")],
    "coordinate_max": [max(a[k] for a in atoms) for k in ("x", "y", "z")],
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

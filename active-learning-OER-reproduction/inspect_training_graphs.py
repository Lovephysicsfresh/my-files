from pathlib import Path
from collections import Counter
import json

ROOT = Path(__file__).resolve().parent / "OER2022" / "initial_train" / "graph"
files = sorted(ROOT.glob("*.pdb"))
assert files, f"No graph PDB files found in {ROOT}"

reports = []
for path in files:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms = [x for x in lines if x.startswith("HETATM")]
    conect = [x for x in lines if x.startswith("CONECT")]
    edges = []
    for line in conect:
        vals = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
        edges.extend((vals[0], v) for v in vals[1:])
    reports.append({
        "file": path.name,
        "atoms": len(atoms),
        "directed_edges": len(edges),
        "elements": dict(Counter(x[76:78].strip() for x in atoms)),
    })

atom_counts = sorted(set(x["atoms"] for x in reports))
edge_counts = sorted(set(x["directed_edges"] for x in reports))
print("files:", len(files))
print("atom_counts:", atom_counts)
print("directed_edge_counts:", edge_counts)
print("first:", json.dumps(reports[:3], indent=2))
assert atom_counts == [336], atom_counts
assert len(files) == 30 * 20, len(files)

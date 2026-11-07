from pathlib import Path
from collections import Counter
import hashlib

ROOT = Path(__file__).resolve().parent / "OER2022" / "initial_train" / "graph"
files = sorted(ROOT.glob("*.pdb"))

def signature(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms = [line for line in lines if line.startswith("HETATM")]
    edges = [line for line in lines if line.startswith("CONECT")]
    element_counts = tuple(sorted(Counter(line[76:78].strip() for line in atoms).items()))
    coords = hashlib.sha256("\n".join(atoms).encode()).hexdigest()
    edge_hash = hashlib.sha256("\n".join(edges).encode()).hexdigest()
    return element_counts, coords, edge_hash

sig_counts = Counter(signature(f) for f in files)
print("files", len(files))
print("distinct signatures", len(sig_counts))
print("signature multiplicities", sorted(sig_counts.values(), reverse=True)[:20])
for f in files[:30]:
    print(f.name, signature(f)[0], signature(f)[1][:12], signature(f)[2][:12])

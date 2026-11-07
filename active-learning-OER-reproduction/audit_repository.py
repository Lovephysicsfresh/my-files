from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent / "OER2022"
TRAIN_FILES = ["initial_train/OER_train_initial.csv", "final_train/OER_train_final.csv"]
SCREEN_FILES = ["initial_screen/OER_screen_initial.csv", "final_screen/OER_screen_final.csv"]

for rel in TRAIN_FILES:
    df = pd.read_csv(ROOT / rel)
    atom_cols = [c for c in df.columns if c.startswith("atom_")]
    xrd_cols = [c for c in df.columns if c.startswith("intensity_") or c.startswith("position_")]
    assert len(df) in (30, 40), (rel, df.shape)
    assert len(atom_cols) == 10, (rel, atom_cols)
    assert len(xrd_cols) == 14, (rel, xrd_cols)
    assert (df[atom_cols].sum(axis=1) - 2).abs().lt(1e-8).all(), rel
    assert not df.isna().any().any(), rel
    assert (df[xrd_cols].to_numpy() >= 0).all(), rel
    print(f"{rel}: PASS rows={len(df)}, atom_features={len(atom_cols)}, xrd_features={len(xrd_cols)}")

for rel in SCREEN_FILES:
    df = pd.read_csv(ROOT / rel)
    atom_cols = [c for c in df.columns if c.startswith("atom_")]
    assert len(df) in (10131, 10141), (rel, df.shape)
    assert len(atom_cols) == 10
    assert (df[atom_cols].sum(axis=1) - 2).abs().lt(1e-8).all(), rel
    assert (df["potential"] == 0).all(), rel
    print(f"{rel}: PASS rows={len(df)}, candidate_potential_placeholder=0")

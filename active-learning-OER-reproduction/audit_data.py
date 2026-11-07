from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent / "OER2022"
A_COLS = ["atom_La", "atom_Ba", "atom_Sr", "atom_Pr", "atom_Ca"]
B_COLS = ["atom_Mn", "atom_Fe", "atom_Co", "atom_Ni", "atom_Cu"]
ATOM_COLS = A_COLS + B_COLS
XRD_COLS = [f"intensity_{i}" for i in range(7)] + [f"position_{i}" for i in range(7)]


def train(rel, n):
    df = pd.read_csv(ROOT / rel)
    assert len(df) == n
    assert not df[["id", "name", "potential"] + ATOM_COLS + XRD_COLS].isna().any().any()
    assert ((df[A_COLS].sum(axis=1) - 1).abs() < 1e-8).all()
    assert ((df[B_COLS].sum(axis=1) - 1).abs() < 1e-8).all()
    assert ((df[ATOM_COLS] >= 0) & (df[ATOM_COLS] <= 1)).all().all()
    assert ((df[XRD_COLS] >= 0) & (df[XRD_COLS] <= 1)).all().all()
    assert (df.potential > 0).all()
    print(f"PASS {rel}: rows={len(df)}, missing=0, A_sum=1, B_sum=1, XRD=[0,1]")


def screen(rel, n):
    df = pd.read_csv(ROOT / rel)
    assert len(df) == n
    assert ((df[A_COLS].sum(axis=1) - 1).abs() < 1e-8).all()
    assert ((df[B_COLS].sum(axis=1) - 1).abs() < 1e-8).all()
    assert (df.potential == 0).all()
    print(f"PASS {rel}: rows={len(df)}, A_sum=1, B_sum=1, potential_placeholder=0")


train("initial_train/OER_train_initial.csv", 30)
train("final_train/OER_train_final.csv", 40)
screen("initial_screen/OER_screen_initial.csv", 10131)
screen("final_screen/OER_screen_final.csv", 10141)
print("ALL DATA AUDITS PASSED")

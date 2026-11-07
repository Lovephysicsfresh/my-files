"""第二步：把材料画出来。

做三件事：
1. 画出 LaMnO3 的 336 个原子和它们之间的键（2x2x2 超胞的 x-y 投影）
2. 把 14 个 XRD 数字还原成 7 个峰
3. 打印模型真正看到的那 4 个数

运行：.\run.bat step_08_visualize.py
产出：figures/step08_graph.png、figures/step08_xrd.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step_03_build_graph import (
    ELECTRONEGATIVITY,
    GRAPH_DIR,
    IONIC_RADIUS,
    parse_pdb,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
index_of = {a["serial"]: i for i, a in enumerate(atoms)}

print("=" * 60)
print("一、这个材料有多少原子")
print("=" * 60)
for site in ("A", "B", "O"):
    n = sum(1 for a in atoms if a["site"] == site)
    print(f"  {site} 位原子数: {n}")
print(f"  合计: {len(atoms)}")

print()
print("=" * 60)
print("二、模型看到的前 6 个原子长什么样")
print("=" * 60)
print(f"{'元素':<6}{'位点':<6}{'半径(Å)':>8}{'电负性':>9}{'A位?':>6}{'B位?':>6}")
for a in atoms[:6]:
    r = IONIC_RADIUS[a["element"]]
    chi = ELECTRONEGATIVITY[a["element"]]
    is_a = 1.0 if a["site"] == "A" else 0.0
    is_b = 1.0 if a["site"] == "B" else 0.0
    print(f"{a['element']:<6}{a['site']:<6}{r:>8.3f}{chi:>9.2f}{is_a:>6.0f}{is_b:>6.0f}")
print("  ...（共 336 行，每行就是这样一个 4 维向量）")

# ---------------- 图1：原子和键 ----------------
xy = np.array([[a["xyz"][0], a["xyz"][1]] for a in atoms])
site = np.array([a["site"] for a in atoms])

fig, ax = plt.subplots(figsize=(8, 7))

for u, v in conect:
    i, j = index_of[u], index_of[v]
    ax.plot(
        [xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
        color="lightgray", linewidth=0.4, zorder=1,
    )

colors = {"A": "#d62728", "B": "#1f77b4", "O": "#7f7f7f"}
sizes = {"A": 60, "B": 40, "O": 18}
for s in ("O", "B", "A"):
    m = site == s
    ax.scatter(
        xy[m, 0], xy[m, 1], s=sizes[s], c=colors[s],
        label=f"{s} site ({m.sum()})", zorder=2, edgecolors="white", linewidths=0.3,
    )

ax.set_title("LaMnO3 supercell, x-y projection (336 atoms, 1296 bonds)")
ax.set_xlabel("x (Angstrom)")
ax.set_ylabel("y (Angstrom)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "step08_graph.png", dpi=150)
plt.close(fig)

# ---------------- 图2：XRD ----------------
df = pd.read_csv(ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
row = df[df["id"] == "P01"].iloc[0]
inten = row[[f"intensity_{i}" for i in range(7)]].to_numpy(float)
pos = row[[f"position_{i}" for i in range(7)]].to_numpy(float)
two_theta = pos * 90.0

print()
print("=" * 60)
print("三、14 个 XRD 数字还原")
print("=" * 60)
print(f"{'峰':<4}{'归一化位置':>12}{'实际2θ(°)':>12}{'相对强度':>10}")
for i in range(7):
    print(f"{i:<4}{pos[i]:>12.4f}{two_theta[i]:>12.2f}{inten[i]:>10.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.vlines(two_theta, 0, inten, color="#1f77b4", linewidth=3)
for i, (x, y) in enumerate(zip(two_theta, inten)):
    ax.text(x, y + 0.04, f"#{i}", ha="center", fontsize=9)
ax.set_xlim(10, 90)
ax.set_ylim(0, 1.25)
ax.set_xlabel("2-theta (degree)")
ax.set_ylabel("relative intensity")
ax.set_title("LaMnO3: 7 peaks (position/90 and intensity/max)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "step08_xrd.png", dpi=150)
plt.close(fig)

print()
print(f"图已保存: {FIG / 'step08_graph.png'}")
print(f"图已保存: {FIG / 'step08_xrd.png'}")

"""I-02 (#2): 二重井戸ポテンシャルの可視化と、障壁高さ→メモリ保持時間の関係。

実行:  uv run python scripts/i02_double_well.py
出力:  outputs/figures/i02_potential.png, outputs/figures/i02_half_life.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面なしでPNGに保存する
import matplotlib.pyplot as plt
import numpy as np

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import half_life_s, potential, reverse_barrier_ev, stationary_points

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- モデルパラメータ（アゾベンゼン程度を想定した例示値） ---
BARRIER_EV = 1.2  # 対称時の障壁高さ E_b [eV]
ASYMMETRY_EV = 0.5  # 状態B（cis想定）が状態A（trans想定）より不安定な度合い [eV]

# 時間スケールの目盛り [s]
TIME_MARKS = {"1 s": 1.0, "1 h": 3600.0, "1 day": 86400.0, "1 year": 3.156e7}


def plot_potential() -> float:
    x = np.linspace(-1.6, 1.6, 400)  # 反応座標（無次元）
    v = potential(x, BARRIER_EV, ASYMMETRY_EV)
    pts = stationary_points(BARRIER_EV, ASYMMETRY_EV)
    de = reverse_barrier_ev(BARRIER_EV, ASYMMETRY_EV)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, v, color="black")
    for name, label in [("min_A", "state A (stable, '0')"), ("min_B", "state B (metastable, '1')"),
                        ("barrier_top", "barrier top")]:
        xi = pts[name]
        ax.plot(xi, potential(xi, BARRIER_EV, ASYMMETRY_EV), "o")
        ax.annotate(label, (xi, potential(xi, BARRIER_EV, ASYMMETRY_EV)),
                    textcoords="offset points", xytext=(5, 8), fontsize=8)
    ax.set_xlabel("reaction coordinate x [dimensionless]")
    ax.set_ylabel("V(x) [eV]")
    ax.set_ylim(-0.6, 2.0)
    ax.set_title(f"Double-well switch: E_b={BARRIER_EV} eV, delta={ASYMMETRY_EV} eV\n"
                 f"barrier B->A: dE = {de:.3f} eV")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i02_potential.png", dpi=150)
    plt.close(fig)
    return de


def plot_half_life(de_model: float) -> None:
    de = np.linspace(0.4, 1.4, 200)  # 障壁高さ [eV]
    temps = [273.15, ROOM_TEMPERATURE_K, 353.15]  # 0, 25, 80 °C [K]

    fig, ax = plt.subplots(figsize=(6, 4))
    for t in temps:
        ax.semilogy(de, half_life_s(de, t), label=f"T = {t - 273.15:.0f} °C")
    for label, sec in TIME_MARKS.items():
        ax.axhline(sec, color="gray", lw=0.5, ls="--")
        ax.text(de[0], sec * 1.3, label, fontsize=7, color="gray")
    ax.axvline(de_model, color="red", lw=0.8, ls=":", label=f"model dE = {de_model:.2f} eV")
    ax.set_xlabel("barrier dE [eV]")
    ax.set_ylabel("half-life t_1/2 [s]")
    ax.set_title("Memory retention time vs. barrier (Eyring)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i02_half_life.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    de_model = plot_potential()
    plot_half_life(de_model)

    print(f"モデル: E_b = {BARRIER_EV} eV, delta = {ASYMMETRY_EV} eV")
    print(f"準安定状態 B → A の障壁 dE = {de_model:.3f} eV\n")
    print(f"{'dE [eV]':>8} | {'t1/2 @25°C':>14}")
    print("-" * 26)
    for de in [0.6, 0.8, 1.0, 1.1, round(de_model, 3)]:
        t = float(half_life_s(de, ROOM_TEMPERATURE_K))
        print(f"{de:>8.3f} | {format_time(t):>14}")
    print(f"\n図を保存: {OUT_DIR}")


def format_time(sec: float) -> str:
    for unit, s in [("years", 3.156e7), ("days", 86400.0), ("h", 3600.0), ("min", 60.0), ("s", 1.0)]:
        if sec >= s:
            return f"{sec / s:.3g} {unit}"
    return f"{sec:.3g} s"


if __name__ == "__main__":
    main()

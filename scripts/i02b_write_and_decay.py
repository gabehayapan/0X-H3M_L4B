"""I-02b: 光でビットを書き込み、暗所で熱により記憶が消えていく様子をシミュレートする。

実行:  uv run python scripts/i02b_write_and_decay.py
出力:  outputs/figures/i02b_write_and_decay.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import eyring_rate, forward_barrier_ev, half_life_s, reverse_barrier_ev
from chemcomp.kinetics import Rates, simulate_segment, time_to_threshold

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- 地形（I-02 と同じ例示値） ---
BARRIER_EV = 1.2  # 対称時の障壁高さ E_b [eV]
ASYMMETRY_EV = 0.5  # 形Bの不安定さ [eV]

# --- 光（例示値。本来は σ·Φ·光子フラックスで決まる） ---
K_PHOTO_AB = 1.0  # 光による A→B の速度 [1/s]（形Aがよく吸収する色の光を想定）
K_PHOTO_BA = 0.1  # 光による B→A の速度 [1/s]（形Bも少し吸収してしまうため逆反応も起きる）
LIGHT_ON_S = 10.0  # 光を当てる時間 [s]

READ_THRESHOLD = 0.5  # p_B がこれ以上なら「1」と読む [無次元]
T_DARK_END_S = 1.0e5  # 暗所での観察時間 [s]（約28時間）
TEMPS_K = [ROOM_TEMPERATURE_K, 323.15]  # 25 °C, 50 °C [K]


def thermal_rates(temperature_k: float) -> Rates:
    """熱だけによる速度（暗所）。"""
    k_ab = float(eyring_rate(forward_barrier_ev(BARRIER_EV, ASYMMETRY_EV), temperature_k))
    k_ba = float(eyring_rate(reverse_barrier_ev(BARRIER_EV, ASYMMETRY_EV), temperature_k))
    return Rates(k_ab=k_ab, k_ba=k_ba)


def run(temperature_k: float):
    dark = thermal_rates(temperature_k)
    light = Rates(k_ab=dark.k_ab + K_PHOTO_AB, k_ba=dark.k_ba + K_PHOTO_BA)

    # 区間1: 光ON（全分子が形A = ビット0 から開始）
    t_on = np.concatenate([[0.0], np.logspace(-3, np.log10(LIGHT_ON_S), 200)])  # 対数軸用に対数間隔
    p_on = simulate_segment(0.0, light, t_on)

    # 区間2: 光OFF（区間1の最後の値から開始）。対数軸で見るので対数間隔で点を取る
    t_off = np.concatenate([[0.0], np.logspace(-2, np.log10(T_DARK_END_S), 400)])
    p_off = simulate_segment(p_on[-1], dark, t_off)

    t_all = np.concatenate([t_on, LIGHT_ON_S + t_off[1:]])
    p_all = np.concatenate([p_on, p_off[1:]])
    t_read = time_to_threshold(p_on[-1], dark, READ_THRESHOLD)
    return t_all, p_all, dark, light, p_on[-1], t_read


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    for temp in TEMPS_K:
        t, p, dark, light, p_written, t_read = run(temp)
        label = f"{temp - 273.15:.0f} °C"
        ax.semilogx(t, p, label=label)
        ax.axvline(LIGHT_ON_S + t_read, ls=":", lw=0.8, color=ax.lines[-1].get_color())

        print(f"=== {label} ===")
        print(f"  熱の速度   A→B: {dark.k_ab:.2e} 1/s   B→A: {dark.k_ba:.2e} 1/s")
        print(f"  光照射中の到達点（光定常状態） p_B = {light.p_b_equilibrium:.3f}")
        print(f"  {LIGHT_ON_S:.0f} s 照射後の書き込み率   p_B = {p_written:.3f}")
        t_half = float(half_life_s(reverse_barrier_ev(BARRIER_EV, ASYMMETRY_EV), temp))
        print(f"  半減期 t1/2（I-02）         = {t_half:,.0f} s")
        print(f"  「1」と読めなくなるまで    = {t_read:,.0f} s  (p_B < {READ_THRESHOLD})")
        print(f"  暗所の最終平衡             p_B = {dark.p_b_equilibrium:.1e}\n")

    ax.axvspan(1e-2, LIGHT_ON_S, color="gold", alpha=0.25, label="light ON (write)")
    ax.axhline(READ_THRESHOLD, color="gray", ls="--", lw=0.8, label=f"read threshold = {READ_THRESHOLD}")
    ax.set_xlim(1e-2, LIGHT_ON_S + T_DARK_END_S)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("time [s] (log scale)")
    ax.set_ylabel("fraction in state B, p_B [-]")
    ax.set_title("Write with light, then thermal decay in the dark\n(dotted: time when bit can no longer be read as '1')")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i02b_write_and_decay.png", dpi=150)
    plt.close(fig)
    print(f"図を保存: {OUT_DIR / 'i02b_write_and_decay.png'}")


if __name__ == "__main__":
    main()

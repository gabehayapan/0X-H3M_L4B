"""M1-05: 書き込み → 保持 → 消去 → 再書き込み の1サイクルをシミュレートする。

- 書き込み: 紫外光（365 nm）。形Aがよく吸収し、A→B が進む
- 消去    : 可視光（450 nm）。形Bがよく吸収し、B→A が進む（熱で待つより速く消せる）
- 保持    : 暗所。熱による B→A だけが働く（I-02 の障壁から計算）

⚠️ ε・Φ は説明用の例示値（I-03 第2版の架空の分子 X と同程度。実在分子の実験値ではない）。

実行:  uv run python scripts/m1_05_write_erase_cycle.py
出力:  outputs/figures/m1_05_write_erase_cycle.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import eyring_rate, forward_barrier_ev, reverse_barrier_ev
from chemcomp.kinetics import Rates, Segment, simulate_protocol
from chemcomp.photo import photo_rate_per_s

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- 地形（I-02 と同じ例示値） ---
BARRIER_EV = 1.2  # 対称時の障壁高さ E_b [eV]
ASYMMETRY_EV = 0.5  # 形Bの不安定さ [eV]

# --- 量子収率（波長によらないと近似） ---
PHI_AB = 0.20  # A→B の量子収率 [−]
PHI_BA = 0.30  # B→A の量子収率 [−]

# --- 書き込み光（紫外） ---
WRITE_NM = 365.0  # [nm]
WRITE_MW_CM2 = 10.0  # 光の強さ [mW/cm²]
EPS_A_WRITE = 2.5e4  # 365 nm での形Aのモル吸光係数 [M⁻¹cm⁻¹]（π-π* 吸収で強い）
EPS_B_WRITE = 2.0e3  # 365 nm での形Bのモル吸光係数 [M⁻¹cm⁻¹]

# --- 消去光（可視） ---
ERASE_NM = 450.0  # [nm]
ERASE_MW_CM2 = 30.0  # 光の強さ [mW/cm²]
EPS_A_ERASE = 5.0e2  # 450 nm での形Aのモル吸光係数 [M⁻¹cm⁻¹]（弱い）
EPS_B_ERASE = 1.5e3  # 450 nm での形Bのモル吸光係数 [M⁻¹cm⁻¹]（n-π* 吸収で形Aより強い）

READ_THRESHOLD = 0.5  # p_B がこれ以上なら「1」と読む [−]

# --- 操作の手順: (名前, 長さ [s], 光の種類) ---
PROTOCOL = [
    ("write", 10.0, "uv"),
    ("hold", 60.0, "dark"),
    ("erase", 30.0, "vis"),
    ("hold", 30.0, "dark"),
    ("re-write", 10.0, "uv"),
    ("hold", 60.0, "dark"),
]
COLORS = {"uv": "violet", "vis": "deepskyblue", "dark": None}


def build_rates(temperature_k: float) -> dict[str, Rates]:
    """光の種類ごとの速度定数（熱 ＋ 光）。"""
    k_th_ab = float(eyring_rate(forward_barrier_ev(BARRIER_EV, ASYMMETRY_EV), temperature_k))
    k_th_ba = float(eyring_rate(reverse_barrier_ev(BARRIER_EV, ASYMMETRY_EV), temperature_k))

    def with_light(eps_a: float, eps_b: float, mw_cm2: float, nm: float) -> Rates:
        return Rates(k_ab=k_th_ab + photo_rate_per_s(eps_a, PHI_AB, mw_cm2, nm),
                     k_ba=k_th_ba + photo_rate_per_s(eps_b, PHI_BA, mw_cm2, nm))

    return {
        "dark": Rates(k_ab=k_th_ab, k_ba=k_th_ba),
        "uv": with_light(EPS_A_WRITE, EPS_B_WRITE, WRITE_MW_CM2, WRITE_NM),
        "vis": with_light(EPS_A_ERASE, EPS_B_ERASE, ERASE_MW_CM2, ERASE_NM),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rates = build_rates(ROOM_TEMPERATURE_K)
    segments = [Segment(name, dur, rates[light]) for name, dur, light in PROTOCOL]
    t, p, ends = simulate_protocol(0.0, segments)

    print("光の種類ごとの到達点（十分長く当てたときの p_B）と時定数:")
    for light in ("uv", "vis"):
        r = rates[light]
        print(f"  {light:<4} p_B → {r.p_b_equilibrium:.3f}   時定数 τ = {1.0 / (r.k_ab + r.k_ba):.2f} s")
    print()

    print(f"{'区間':<10} | {'長さ [s]':>8} | {'終わりの p_B':>12} | 読み取り")
    print("-" * 48)
    for (name, dur, _), p_end in zip(PROTOCOL, ends):
        bit = "1" if p_end >= READ_THRESHOLD else "0"
        print(f"{name:<10} | {dur:>8.0f} | {p_end:>12.3f} | {bit}")

    erase_end = ends[2]
    status = "OK" if erase_end < READ_THRESHOLD else "NG"
    print(f"\n消去後の p_B = {erase_end:.3f}（境目 {READ_THRESHOLD} より下なら OK）→ {status}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, p, color="black")
    t0 = 0.0
    for name, dur, light in PROTOCOL:
        if COLORS[light]:
            ax.axvspan(t0, t0 + dur, color=COLORS[light], alpha=0.25)
        ax.text(t0 + dur / 2, 1.02, name, ha="center", fontsize=8)
        t0 += dur
    ax.axhline(READ_THRESHOLD, color="gray", ls="--", lw=0.8)
    ax.text(t0, READ_THRESHOLD + 0.02, "read threshold", ha="right", fontsize=7, color="gray")
    ax.set_xlim(0, t0)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction in state B, p_B [-]")
    ax.set_title("1-bit chemical memory: write (UV) / hold (dark) / erase (visible) / re-write", pad=18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m1_05_write_erase_cycle.png", dpi=150)
    plt.close(fig)
    print(f"図を保存: {OUT_DIR / 'm1_05_write_erase_cycle.png'}")


if __name__ == "__main__":
    main()

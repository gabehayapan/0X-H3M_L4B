"""I-03 (#3) 第2版: 6指標で「架空の分子」2つを採点し、内訳を比べる。

⚠️ 分子 X / Y の数値は説明用の架空の値（実在の分子の実験値ではない）。

書き込み率 p_PSS と書き込み時間 τ_w は、ε（吸収しやすさ）・Φ（量子収率）・光の強さから計算する。
実行:  uv run python scripts/i03_score_v2.py
出力:  outputs/figures/i03_score_v2.png
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import half_life_s
from chemcomp.photo import photo_rate_per_s
from chemcomp.score import readout_on_off, score_switch

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- 書き込み用の光（動作条件。分子の性質ではない） ---
WRITE_WAVELENGTH_NM = 365.0  # 紫外光 [nm]
WRITE_INTENSITY_MW_CM2 = 10.0  # 光の強さ [mW/cm²]（実験室のUV LED程度）


@dataclass
class Molecule:
    name: str
    barrier_ev: float  # 熱で B→A に戻る山の高さ ΔE‡ [eV]
    eps_a: float  # 書き込み波長での形Aのモル吸光係数 [M⁻¹cm⁻¹]
    eps_b: float  # 書き込み波長での形Bのモル吸光係数 [M⁻¹cm⁻¹]
    phi_ab: float  # 量子収率 A→B [−]
    phi_ba: float  # 量子収率 B→A [−]
    delta_lambda_nm: float  # 2つの形の吸収極大の差 [nm]
    eps_ratio_read: float  # 読み取り波長での ε_B / ε_A [−]
    cycles: float | None  # 疲労耐性 [回]（None = 不明）


MOLECULES = [
    Molecule("X: fast but fragile", 0.90, 2.5e4, 2.0e3, 0.20, 0.30, 90.0, 6.0, 3e2),
    Molecule("Y: slow but robust", 0.92, 1.5e4, 2.0e3, 0.25, 0.10, 120.0, 8.0, 2e4),
    Molecule("X (fatigue unknown)", 0.90, 2.5e4, 2.0e3, 0.20, 0.30, 90.0, 6.0, None),
]
PART_LABELS = ["t_half", "p_pss", "delta_lambda", "on_off", "write_time", "cycles"]


def evaluate(m: Molecule):
    k_ab = photo_rate_per_s(m.eps_a, m.phi_ab, WRITE_INTENSITY_MW_CM2, WRITE_WAVELENGTH_NM)  # [1/s]
    k_ba = photo_rate_per_s(m.eps_b, m.phi_ba, WRITE_INTENSITY_MW_CM2, WRITE_WAVELENGTH_NM)  # [1/s]
    p_pss = k_ab / (k_ab + k_ba)  # = ε_A·Φ_AB / (ε_A·Φ_AB + ε_B·Φ_BA)
    write_time = 1.0 / (k_ab + k_ba)  # 光定常状態に近づく時定数 [s]（熱の速さは光より桁違いに小さいので無視）
    t_half = float(half_life_s(m.barrier_ev, ROOM_TEMPERATURE_K))
    on_off = readout_on_off(p_pss, m.eps_ratio_read)
    result = score_switch(t_half, p_pss, m.delta_lambda_nm, on_off=on_off,
                          write_time_s=write_time, cycles=m.cycles)
    raw = {"t_half": f"{t_half:.0f} s", "p_pss": f"{p_pss:.2f}", "delta_lambda": f"{m.delta_lambda_nm:.0f} nm",
           "on_off": f"{on_off:.1f}", "write_time": f"{write_time:.2f} s",
           "cycles": "unknown" if m.cycles is None else f"{m.cycles:.0f}"}
    return result, raw


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"書き込み光: {WRITE_WAVELENGTH_NM:.0f} nm, {WRITE_INTENSITY_MW_CM2:.0f} mW/cm²  (25 °C)\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(MOLECULES)
    x = np.arange(len(PART_LABELS))
    for i, m in enumerate(MOLECULES):
        r, raw = evaluate(m)
        print(f"=== {m.name} ===  総合スコア S = {r.total:.2f}")
        for name in PART_LABELS:
            d = r.parts.get(name)
            d_str = "  -  " if d is None else f"{d:.2f}"
            print(f"  {name:<13} 値 = {raw[name]:>9}   d = {d_str}")
        print(f"  足切り: {', '.join(r.failed) or 'なし'}   不明: {', '.join(r.missing) or 'なし'}\n")
        heights = [r.parts.get(n, np.nan) for n in PART_LABELS]
        ax.bar(x + (i - (len(MOLECULES) - 1) / 2) * width, heights, width,
               label=f"{m.name}  (S = {r.total:.2f})")

    ax.set_xticks(x, PART_LABELS)
    ax.set_ylabel("desirability d [-]")
    ax.set_ylim(0, 1.3)
    ax.set_title("Score breakdown, use case A (hypothetical molecules)\n(missing bar = value unknown)")
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i03_score_v2.png", dpi=150)
    plt.close(fig)
    print(f"図を保存: {OUT_DIR / 'i03_score_v2.png'}")


if __name__ == "__main__":
    main()

"""I-03 (#3) 第1版: 山の高さ ΔE‡ を変えたとき、用途Aのスコアがどう変わるか。

書き込み率と波長の分離は固定し、山の高さだけを動かす。
実行:  uv run python scripts/i03_score_window.py
出力:  outputs/figures/i03_score_window.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import half_life_s
from chemcomp.score import score_switch

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

P_PSS = 0.91  # 書き込み率 [-]（I-02b の光定常状態の値）
DELTA_LAMBDA_NM = 80.0  # 波長の分離 [nm]（例示値）
TEMPS_K = [ROOM_TEMPERATURE_K, 308.15]  # 25 °C, 35 °C [K]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    barriers = np.linspace(0.70, 1.05, 351)  # 山の高さ ΔE‡ [eV]

    fig, ax = plt.subplots(figsize=(7, 4))
    for temp in TEMPS_K:
        totals = []
        for de in barriers:
            r = score_switch(float(half_life_s(de, temp)), P_PSS, DELTA_LAMBDA_NM)
            totals.append(r.total)
        totals = np.array(totals)
        label = f"T = {temp - 273.15:.0f} °C"
        ax.plot(barriers, totals, label=label)

        ok = barriers[totals > 0.0]
        best = barriers[totals >= totals.max() - 1e-9]
        print(f"=== {label} ===")
        print(f"  スコア > 0（足切りを通過）: ΔE‡ = {ok.min():.3f} 〜 {ok.max():.3f} eV")
        print(f"  最高スコア {totals.max():.3f}  : ΔE‡ = {best.min():.3f} 〜 {best.max():.3f} eV "
              f"(幅 {best.max() - best.min():.3f} eV)\n")

    print("例（25 °C）:")
    print(f"  {'ΔE‡ [eV]':>9} | {'t1/2':>10} | {'d_t½':>5} | {'d_pPSS':>6} | {'d_Δλ':>5} | {'S':>5} | 足切り")
    for de in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        t = float(half_life_s(de, ROOM_TEMPERATURE_K))
        r = score_switch(t, P_PSS, DELTA_LAMBDA_NM)
        print(f"  {de:>9.2f} | {t:>8.3g} s | {r.parts['t_half']:>5.2f} | {r.parts['p_pss']:>6.2f} | "
              f"{r.parts['delta_lambda']:>5.2f} | {r.total:>5.2f} | {', '.join(r.failed) or '-'}")

    ax.set_xlabel("barrier dE [eV]")
    ax.set_ylabel("score S [-]")
    ax.set_title(f"Use case A (temporary memory) score vs. barrier\n"
                 f"(p_PSS = {P_PSS}, d_lambda = {DELTA_LAMBDA_NM:.0f} nm fixed)")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i03_score_window.png", dpi=150)
    plt.close(fig)
    print(f"\n図を保存: {OUT_DIR / 'i03_score_window.png'}")


if __name__ == "__main__":
    main()

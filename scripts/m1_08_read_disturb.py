"""M1-08: 読み取りの副作用（read disturb）。何回読むと 0/1 が壊れるか。

読み取り = 読み取り波長の弱い光を短時間当てて吸光度を測る。光を吸収した分子は Φ の確率で形が変わる。
3つの読み取り波長を比べる:
    365 nm: 書き込みと同じ紫外光。形Aがよく吸収 → 形Bが増える向き（「0」が壊れる）
    450 nm: 消去と同じ可視光。形Bがやや強く吸収 → 形Aに戻る向き（「1」が壊れる）
    520 nm: 形Bだけがよく吸収し、形Aはほとんど吸収しない → コントラストは高いが「1」が壊れる向き

⚠️ ε・Φ は説明用の例示値（M1-05 と同じ架空の分子。520 nm の ε は M1-08 で追加した仮の値）。

実行:  uv run python scripts/m1_08_read_disturb.py
出力:  outputs/figures/m1_08_read_disturb.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemcomp.components.memory_bit import LightPulse, ReadoutConfig, SwitchParams, dark, read_bit, simulate_bit
from chemcomp.components.read_disturb import read_cycle_map, reads_until_flip
from chemcomp.constants import ROOM_TEMPERATURE_K

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

# --- 分子（M1-05 と同じ。520 nm を追加） ---
PARAMS = SwitchParams(
    barrier_ev=1.2,  # 対称時の障壁高さ E_b [eV]
    asymmetry_ev=0.5,  # 形Bの不安定さ [eV]
    phi_ab=0.20,  # 量子収率 A→B [−]
    phi_ba=0.30,  # 量子収率 B→A [−]
    epsilon_a={365.0: 2.5e4, 450.0: 5.0e2, 520.0: 5.0e1},  # 形Aのモル吸光係数 [M⁻¹cm⁻¹]
    epsilon_b={365.0: 2.0e3, 450.0: 1.5e3, 520.0: 6.0e2},  # 形Bのモル吸光係数 [M⁻¹cm⁻¹]
)
T = ROOM_TEMPERATURE_K  # 25 °C [K]

# --- 0 と 1 を用意する手順（M1-05 と同じ光） ---
WRITE = LightPulse(365.0, 10.0, 10.0, "write")  # 紫外 10 mW/cm² × 10 s
ERASE = LightPulse(450.0, 30.0, 30.0, "erase")  # 可視 30 mW/cm² × 30 s

# --- 読み取り光（弱く・短く） ---
READ_INTENSITY_MW_CM2 = 0.1  # 書き込み光の 1/100 [mW/cm²]
READ_DURATION_S = 0.1  # 1回の読み取りにかける時間 [s]
READ_WAVELENGTHS_NM = [365.0, 450.0, 520.0]
THRESHOLD = 0.5  # p_B ≥ 0.5 を「1」と読む [−]


def prepared_states() -> tuple[float, float]:
    """書き込み直後（1）と、書き込み→消去の後（0）の p_B。"""
    trace = simulate_bit(PARAMS, [WRITE, ERASE], T, p_b0=0.0)
    return trace.pulse_end_p_b[0], trace.pulse_end_p_b[1]


def fmt_reads(n: int | None) -> str:
    return "壊れない" if n is None else f"{n:,} 回"


def fmt_time(n: int | None) -> str:
    return "-" if n is None else f"{n * READ_DURATION_S:,.0f} s"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1, p0 = prepared_states()
    print(f"準備した状態: 「1」 p_B = {p1:.3f}（書き込み直後） / 「0」 p_B = {p0:.3f}（消去後）")
    print(f"読み取り光: {READ_INTENSITY_MW_CM2} mW/cm² × {READ_DURATION_S} s / 回（連続して読む）, 25 °C\n")

    # 比較の基準: 光を当てずに同じ時間待つだけ（熱だけで消える）
    dark_cycle = read_cycle_map(PARAMS, dark(READ_DURATION_S), T)
    n_dark_1 = reads_until_flip(dark_cycle, p1, THRESHOLD)
    print(f"[基準] 光なしで同じ時間待つと「1」が消えるまで: {fmt_reads(n_dark_1)} 分の時間 ({fmt_time(n_dark_1)})\n")

    header = (f"{'λ [nm]':>6} | {'近づく先 p*':>10} | {'1回の変化(1)':>12} | {'1回の変化(0)':>12} | "
              f"{'「1」が壊れる':>12} | {'「0」が壊れる':>12} | {'ON/OFF':>6}")
    print(header)
    print("-" * len(header))

    reads = np.unique(np.logspace(0, 5, 300).astype(int))  # 読み取り回数 [回]
    fig, axes = plt.subplots(1, len(READ_WAVELENGTHS_NM), figsize=(11, 3.8), sharey=True)
    for ax, lam in zip(axes, READ_WAVELENGTHS_NM):
        cycle = read_cycle_map(PARAMS, LightPulse(lam, READ_INTENSITY_MW_CM2, READ_DURATION_S, "read"), T)
        p_star = cycle.fixed_point
        n1, n0 = reads_until_flip(cycle, p1, THRESHOLD), reads_until_flip(cycle, p0, THRESHOLD)
        readout = ReadoutConfig(lam, THRESHOLD)
        on_off = read_bit(PARAMS, p1, readout).signal / read_bit(PARAMS, p0, readout).signal
        on_off = max(on_off, 1.0 / on_off)  # 信号が増えても減っても「何倍違うか」で表す
        print(f"{lam:>6.0f} | {p_star:>10.3f} | {cycle(p1) - p1:>+12.2e} | {cycle(p0) - p0:>+12.2e} | "
              f"{fmt_reads(n1):>12} | {fmt_reads(n0):>12} | {on_off:>6.1f}")

        # p_n = p* + α^n (p_0 − p*)
        for p_start, label, color in [(p1, "bit 1", "tab:red"), (p0, "bit 0", "tab:blue")]:
            ax.semilogx(reads, p_star + cycle.alpha ** reads * (p_start - p_star), color=color, label=label)
        ax.semilogx(reads, dark_cycle.fixed_point + dark_cycle.alpha ** reads * (p1 - dark_cycle.fixed_point),
                    color="tab:red", ls=":", label="bit 1, dark only")
        ax.axhline(THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.set_title(f"read at {lam:.0f} nm (ON/OFF = {on_off:.1f})", fontsize=9)
        ax.set_xlabel("number of reads")
    axes[0].set_ylabel("fraction in state B, p_B [-]")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Read disturb: {READ_INTENSITY_MW_CM2} mW/cm² x {READ_DURATION_S} s per read, 25 °C")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m1_08_read_disturb.png", dpi=150)
    plt.close(fig)

    print("\n見方: 「1回の変化」が 0 に近いほど副作用が小さい。「壊れない」は、読み続けても近づく先 p* が")
    print("      同じ側にあるので反転しないという意味。ON/OFF は 0 と 1 の信号が何倍違うか（読み取りやすさ）。")
    print("      「1」が壊れる回数を [基準] と比べると、読み取り光のせいでどれだけ寿命が縮んだかが分かる。")
    print(f"図を保存: {OUT_DIR / 'm1_08_read_disturb.png'}")


if __name__ == "__main__":
    main()

"""1ビット化学メモリ部品（M1-07）。設計は ARCHITECT.md §8（M1-06）を参照。

入力は「光パルスの列」、内部状態は p_B（形Bの割合）、出力は読み取り結果。
計算の中身は既存コードの組み合わせ:
    熱の速さ  ← double_well.eyring_rate（障壁から、I-02）
    光の速さ  ← photo.photo_rate_per_s（ε・Φ・光の強さから、I-03 第2版）
    時間発展  ← kinetics.simulate_protocol（区間をつなぐ、M1-05）
"""

from dataclasses import dataclass, field

import numpy as np

from chemcomp.double_well import eyring_rate, forward_barrier_ev, reverse_barrier_ev
from chemcomp.kinetics import Rates, Segment, simulate_protocol
from chemcomp.photo import photo_rate_per_s

DEFAULT_READ_THRESHOLD = 0.5  # p_B がこれ以上なら「1」と読む [−]


@dataclass(frozen=True)
class LightPulse:
    """1つの操作区間。intensity_mw_cm2 = 0 が暗所（保持）。"""

    wavelength_nm: float  # 波長 [nm]（暗所では使わない）
    intensity_mw_cm2: float  # 光の強さ [mW/cm²]
    duration_s: float  # 長さ [s]
    label: str = ""  # 区間の名前（例: "write", "hold"）。図の注記用

    def __post_init__(self) -> None:
        if self.intensity_mw_cm2 < 0:
            raise ValueError(f"光の強さは 0 以上である必要がある: {self.intensity_mw_cm2}")
        if self.duration_s <= 0:
            raise ValueError(f"区間の長さは正である必要がある: {self.duration_s}")


def dark(duration_s: float, label: str = "hold") -> LightPulse:
    """暗所（保持）の区間を作る。"""
    return LightPulse(wavelength_nm=0.0, intensity_mw_cm2=0.0, duration_s=duration_s, label=label)


@dataclass(frozen=True)
class SwitchParams:
    """分子の性質。"""

    barrier_ev: float  # 対称時の障壁高さ E_b [eV]（I-02 の二重井戸）
    asymmetry_ev: float  # 形Bの不安定さ δ [eV]
    phi_ab: float  # 量子収率 A→B [−]（波長によらないと近似）
    phi_ba: float  # 量子収率 B→A [−]
    epsilon_a: dict[float, float] = field(default_factory=dict)  # 波長 [nm] → 形Aのモル吸光係数 [M⁻¹cm⁻¹]
    epsilon_b: dict[float, float] = field(default_factory=dict)  # 波長 [nm] → 形Bのモル吸光係数 [M⁻¹cm⁻¹]


@dataclass(frozen=True)
class ReadoutConfig:
    """読み取り方。p_B ≥ threshold を「1」とする。"""

    wavelength_nm: float  # 読み取り波長 [nm]
    threshold: float = DEFAULT_READ_THRESHOLD  # [−]


@dataclass
class ReadResult:
    bit: int  # 0 または 1
    p_b: float  # 形Bの割合 [−]
    signal: float  # 読み取り波長での吸光度（全分子が形Aのときを 1 とした相対値）[−]


@dataclass
class Trace:
    t: np.ndarray  # 時刻 [s]
    p_b: np.ndarray  # 形Bの割合 [−]
    pulse_end_p_b: list[float]  # 各パルスの終わりの p_B [−]


def _epsilon(table: dict[float, float], wavelength_nm: float, form: str) -> float:
    """辞書から ε を取り出す。ない波長は黙って 0 にせず KeyError にする。"""
    if wavelength_nm not in table:
        raise KeyError(f"形{form}の ε が波長 {wavelength_nm} nm で定義されていない "
                       f"(定義済み: {sorted(table)})")
    return table[wavelength_nm]


def rates_for_pulse(params: SwitchParams, pulse: LightPulse, temperature_k: float) -> Rates:
    """1つのパルスの間の速度定数（熱 ＋ 光）[1/s]。"""
    k_ab = float(eyring_rate(forward_barrier_ev(params.barrier_ev, params.asymmetry_ev), temperature_k))
    k_ba = float(eyring_rate(reverse_barrier_ev(params.barrier_ev, params.asymmetry_ev), temperature_k))
    if pulse.intensity_mw_cm2 > 0:
        lam, inten = pulse.wavelength_nm, pulse.intensity_mw_cm2
        k_ab += photo_rate_per_s(_epsilon(params.epsilon_a, lam, "A"), params.phi_ab, inten, lam)
        k_ba += photo_rate_per_s(_epsilon(params.epsilon_b, lam, "B"), params.phi_ba, inten, lam)
    return Rates(k_ab=k_ab, k_ba=k_ba)


def simulate_bit(params: SwitchParams, pulses: list[LightPulse], temperature_k: float,
                 p_b0: float = 0.0, points_per_pulse: int = 200) -> Trace:
    """光パルスの列を順に当てたときの p_B(t)。状態を持たない純粋な関数。"""
    if not 0.0 <= p_b0 <= 1.0:
        raise ValueError(f"p_b0 は 0〜1 である必要がある: {p_b0}")
    segments = [Segment(p.label, p.duration_s, rates_for_pulse(params, p, temperature_k)) for p in pulses]
    t, p_b, ends = simulate_protocol(p_b0, segments, points_per_segment=points_per_pulse)
    return Trace(t=t, p_b=p_b, pulse_end_p_b=ends)


def read_bit(params: SwitchParams, p_b: float, readout: ReadoutConfig) -> ReadResult:
    """理想的な読み取り（状態を変えない）。副作用のある読み取りは LightPulse で表す（M1-08）。"""
    eps_a = _epsilon(params.epsilon_a, readout.wavelength_nm, "A")
    eps_b = _epsilon(params.epsilon_b, readout.wavelength_nm, "B")
    if eps_a <= 0:
        raise ValueError("読み取り波長での形Aの ε は正である必要がある（信号を ε_A で割るため）")
    signal = (p_b * eps_b + (1.0 - p_b) * eps_a) / eps_a
    return ReadResult(bit=int(p_b >= readout.threshold), p_b=p_b, signal=signal)


class MemoryBit:
    """状態（p_B）を持つ薄いラッパー。回路やビジュアライザから1ステップずつ動かす用。"""

    def __init__(self, params: SwitchParams, temperature_k: float, readout: ReadoutConfig,
                 p_b0: float = 0.0) -> None:
        if not 0.0 <= p_b0 <= 1.0:
            raise ValueError(f"p_b0 は 0〜1 である必要がある: {p_b0}")
        self.params = params
        self.temperature_k = temperature_k  # 温度 [K]
        self.readout = readout
        self.p_b = p_b0  # 今の状態 [−]

    def apply(self, pulses: list[LightPulse]) -> Trace:
        """光を当てて状態を進める。時刻は今回の呼び出しの開始を 0 とする。"""
        trace = simulate_bit(self.params, pulses, self.temperature_k, p_b0=self.p_b)
        self.p_b = float(trace.p_b[-1])
        return trace

    def read(self) -> ReadResult:
        """今の状態を読む（副作用なし）。"""
        return read_bit(self.params, self.p_b, self.readout)

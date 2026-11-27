"""読み取りの副作用（read disturb）の計算（M1-08）。

実際の読み取りは「読み取り波長の弱い光を短時間当てて吸光度を測る」こと。
光を吸収した分子は量子収率 Φ の確率で形が変わるので、読むたびに状態が少しずつ動く。

考え方:
    速度が一定の区間の解析解（kinetics.analytic_segment）
        p' = p_eq + (p − p_eq)·e^(−K·t),   K = k_AB + k_BA
    は p について 1次式（アフィン写像） p' = α·p + β になる:
        α = e^(−K·t),   β = p_eq·(1 − α)
    「読み取り光 → 暗所の待ち時間」の1サイクルも、2つの1次式の合成なので1次式になる。
    よって n 回読んだ後は
        p_n − p* = α^n · (p_0 − p*),   p* = β / (1 − α)   （読み続けると近づく値）
    となり、「何回読むと 0/1 が反転するか」が式で求まる。

注: 区間中の速度定数が一定（光の強さ・温度が一定）と仮定している。
"""

from dataclasses import dataclass

import numpy as np

from chemcomp.components.memory_bit import LightPulse, SwitchParams, dark, rates_for_pulse
from chemcomp.kinetics import Rates


@dataclass(frozen=True)
class AffineMap:
    """p' = alpha·p + beta。1サイクルで p_B がどう動くか。"""

    alpha: float  # 前の状態がどれだけ残るか [−]（0〜1。1 に近いほど副作用が小さい）
    beta: float  # [−]

    def __call__(self, p_b: float) -> float:
        return self.alpha * p_b + self.beta

    def then(self, other: "AffineMap") -> "AffineMap":
        """self を当ててから other を当てる合成写像。"""
        return AffineMap(alpha=other.alpha * self.alpha, beta=other.alpha * self.beta + other.beta)

    @property
    def fixed_point(self) -> float | None:
        """何度も繰り返すと近づく p_B（p* = β / (1 − α)）。α = 1（何も起きない）なら None。"""
        if self.alpha >= 1.0:
            return None
        return self.beta / (1.0 - self.alpha)


def segment_map(rates: Rates, duration_s: float) -> AffineMap:
    """速度が一定の区間を1次式で表す。"""
    k_total = rates.k_ab + rates.k_ba  # [1/s]
    alpha = float(np.exp(-k_total * duration_s))
    return AffineMap(alpha=alpha, beta=rates.p_b_equilibrium * (1.0 - alpha))


def read_cycle_map(params: SwitchParams, read_pulse: LightPulse, temperature_k: float,
                   dark_between_s: float = 0.0) -> AffineMap:
    """読み取り1回（読み取り光 → 暗所で dark_between_s 待つ）の写像。"""
    cycle = segment_map(rates_for_pulse(params, read_pulse, temperature_k), read_pulse.duration_s)
    if dark_between_s > 0:
        wait = dark(dark_between_s)
        cycle = cycle.then(segment_map(rates_for_pulse(params, wait, temperature_k), dark_between_s))
    return cycle


def reads_until_flip(cycle: AffineMap, p_b0: float, threshold: float) -> int | None:
    """読み取りを繰り返したとき、p_B が threshold をまたぐ（0/1 が反転する）最初の回数。

    - 最初のビットは p_b0 から決める（p_b0 ≥ threshold なら「1」）
    - 読み続けても threshold をまたがない（近づく先 p* が同じ側）場合は None（何回読んでも壊れない）
    """
    bit0 = p_b0 >= threshold
    p_star = cycle.fixed_point
    if p_star is None or (p_star >= threshold) == bit0:
        return None
    # p_n − p* = α^n (p_0 − p*) が (threshold − p*) を越える最小の n
    ratio = (threshold - p_star) / (p_b0 - p_star)  # 0 < ratio < 1
    n = int(np.ceil(np.log(ratio) / np.log(cycle.alpha)))
    # 浮動小数の丸めで境目ちょうどになった場合の補正: n 回後に本当に反転しているか確かめる
    p_n = p_star + cycle.alpha ** n * (p_b0 - p_star)
    if (p_n >= threshold) == bit0:
        n += 1
    return max(n, 0)

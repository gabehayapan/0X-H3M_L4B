"""読み取りの副作用（M1-08）の検算テスト。"""

import pytest

from chemcomp.components.memory_bit import LightPulse, MemoryBit, ReadoutConfig, SwitchParams, dark, simulate_bit
from chemcomp.components.read_disturb import AffineMap, read_cycle_map, reads_until_flip, segment_map
from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.kinetics import Rates

# 説明用の架空の分子（M1-05 と同じ地形・量子収率）。520 nm は形Bだけがよく吸収する読み取り波長
PARAMS = SwitchParams(
    barrier_ev=1.2, asymmetry_ev=0.5, phi_ab=0.20, phi_ba=0.30,
    epsilon_a={365.0: 2.5e4, 520.0: 5.0e1},
    epsilon_b={365.0: 2.0e3, 520.0: 6.0e2},
)
T = ROOM_TEMPERATURE_K
# テストを速くするため、わざと強い読み取り光（30 mW/cm² × 0.5 s）にして数十回で壊れるようにする
STRONG_READ = LightPulse(520.0, 30.0, 0.5, "read")


def test_affine_composition():
    # 2つの1次式の合成: f(p) = 0.5p + 0.1, g(p) = 0.8p + 0.2 → g(f(p)) = 0.4p + 0.28
    h = AffineMap(0.5, 0.1).then(AffineMap(0.8, 0.2))
    assert (h.alpha, h.beta) == pytest.approx((0.4, 0.28))
    assert h.fixed_point == pytest.approx(0.28 / 0.6)


def test_segment_map_fixed_point_is_equilibrium():
    # 1区間を繰り返すと、その光の下での平衡（光定常状態）に近づく
    rates = Rates(k_ab=1.0, k_ba=0.1)
    assert segment_map(rates, 0.3).fixed_point == pytest.approx(rates.p_b_equilibrium)


def test_cycle_map_matches_numerical_simulation():
    # 読み取り → 暗所 2 s の1サイクルを、式（1次式）と数値積分で比べる
    cycle = read_cycle_map(PARAMS, STRONG_READ, T, dark_between_s=2.0)
    trace = simulate_bit(PARAMS, [STRONG_READ, dark(2.0)], T, p_b0=0.85)
    assert cycle(0.85) == pytest.approx(trace.pulse_end_p_b[-1], abs=1e-6)


def test_reads_until_flip_matches_repeated_reads():
    # 式で求めた回数 n の1回前まではビットが保たれ、n 回目で反転する
    cycle = read_cycle_map(PARAMS, STRONG_READ, T)
    n = reads_until_flip(cycle, 0.85, 0.5)
    assert n is not None and 5 < n < 200  # 数十回程度になるよう光を選んでいる
    bit = MemoryBit(PARAMS, T, ReadoutConfig(520.0), p_b0=0.85)
    bit.apply([STRONG_READ] * (n - 1))
    assert bit.read().bit == 1
    bit.apply([STRONG_READ])
    assert bit.read().bit == 0


def test_never_flips_when_reads_push_toward_same_side():
    # 520 nm の読み取りは形Bを減らす向き（p* < 0.5）なので、「0」は何回読んでも壊れない
    cycle = read_cycle_map(PARAMS, STRONG_READ, T)
    assert cycle.fixed_point < 0.5
    assert reads_until_flip(cycle, 0.05, 0.5) is None


def test_read_at_write_wavelength_breaks_bit_zero():
    # 書き込みと同じ 365 nm で読むと、形B を増やす向き（p* > 0.5）なので「0」が壊れる
    cycle = read_cycle_map(PARAMS, LightPulse(365.0, 1.0, 0.5, "read"), T)
    assert cycle.fixed_point > 0.5
    assert reads_until_flip(cycle, 0.05, 0.5) is not None
    assert reads_until_flip(cycle, 0.85, 0.5) is None

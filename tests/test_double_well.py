"""物理的な検算テスト。"""

import numpy as np
import pytest

from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import eyring_rate, half_life_s, reverse_barrier_ev, stationary_points


def test_symmetric_barrier_equals_eb():
    # 対称 (delta=0) なら極小は x=±1、頂上は x=0 で、障壁はちょうど E_b
    pts = stationary_points(1.0, 0.0)
    assert pts["min_A"] == pytest.approx(-1.0)
    assert pts["barrier_top"] == pytest.approx(0.0, abs=1e-12)
    assert pts["min_B"] == pytest.approx(1.0)
    assert reverse_barrier_ev(1.0, 0.0) == pytest.approx(1.0)


def test_asymmetry_lowers_reverse_barrier():
    # 状態Bを不安定化すると、B→A の障壁は下がる（メモリ寿命が短くなる）
    assert reverse_barrier_ev(1.0, 0.3) < reverse_barrier_ev(1.0, 0.0)


def test_eyring_prefactor_at_room_temperature():
    # ΔE=0 のとき k = k_B T / h ≈ 6.21e12 1/s (298.15 K)
    assert eyring_rate(0.0, ROOM_TEMPERATURE_K) == pytest.approx(6.21e12, rel=1e-2)


def test_half_life_scaling():
    # 障壁が k_B T*ln(10) 上がるごとに半減期は10倍になる
    from chemcomp.constants import K_B_EV
    t = ROOM_TEMPERATURE_K
    step = K_B_EV * t * np.log(10.0)
    assert half_life_s(1.0 + step, t) / half_life_s(1.0, t) == pytest.approx(10.0)


def test_too_asymmetric_raises():
    with pytest.raises(ValueError):
        stationary_points(1.0, 10.0)

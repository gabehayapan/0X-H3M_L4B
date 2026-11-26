"""2状態速度論の検算テスト。"""

import numpy as np
import pytest

from chemcomp.double_well import forward_barrier_ev, reverse_barrier_ev
from chemcomp.kinetics import Rates, analytic_segment, simulate_segment, time_to_threshold


def test_numeric_matches_analytic():
    # 数値積分が解析解と一致するか
    rates = Rates(k_ab=1.0, k_ba=0.1)
    t = np.linspace(0.0, 10.0, 50)
    assert simulate_segment(0.0, rates, t) == pytest.approx(analytic_segment(0.0, rates, t), abs=1e-6)


def test_photostationary_state():
    # 十分長く光を当てると p_B = k_AB / (k_AB + k_BA) に落ち着く
    rates = Rates(k_ab=1.0, k_ba=0.1)
    p_end = simulate_segment(0.0, rates, np.array([0.0, 100.0]))[-1]
    assert p_end == pytest.approx(1.0 / 1.1, rel=1e-6)


def test_pure_decay_threshold_equals_half_life():
    # 逆反応なし・100%書き込みから 0.5 を切る時間は、半減期 ln2/k と一致する
    k = 1e-3
    assert time_to_threshold(1.0, Rates(k_ab=0.0, k_ba=k), 0.5) == pytest.approx(np.log(2) / k)


def test_barrier_difference_equals_well_difference():
    # 前向き障壁 − 逆向き障壁 = 2つの井戸のエネルギー差（ヘスの法則と同じ考え方）
    from chemcomp.double_well import potential, stationary_points
    eb, d = 1.2, 0.5
    pts = stationary_points(eb, d)
    well_gap = potential(pts["min_B"], eb, d) - potential(pts["min_A"], eb, d)
    assert forward_barrier_ev(eb, d) - reverse_barrier_ev(eb, d) == pytest.approx(well_gap)


# ---------- M1-05: 区間をつなぐシミュレーション ----------

from chemcomp.kinetics import Segment, simulate_protocol  # noqa: E402


def test_protocol_matches_piecewise_analytic():
    # 2区間をつないだ結果が、解析解を2回つないだものと一致する
    light, dark = Rates(k_ab=1.0, k_ba=0.1), Rates(k_ab=0.0, k_ba=0.01)
    _, _, ends = simulate_protocol(0.0, [Segment("w", 5.0, light), Segment("h", 50.0, dark)])
    p1 = analytic_segment(0.0, light, np.array([5.0]))[0]
    p2 = analytic_segment(p1, dark, np.array([50.0]))[0]
    assert ends == pytest.approx([p1, p2], abs=1e-6)


def test_protocol_time_axis_is_continuous():
    seg = Rates(k_ab=1.0, k_ba=1.0)
    t, p, _ = simulate_protocol(0.0, [Segment("a", 2.0, seg), Segment("b", 3.0, seg)], points_per_segment=11)
    assert t[0] == 0.0 and t[-1] == pytest.approx(5.0)
    assert np.all(np.diff(t) > 0)  # 時刻が単調に増える（区間の境目で重複しない）
    assert len(t) == len(p) == 1 + 2 * 10


def test_erase_light_drives_below_threshold():
    # 消去光の到達点が 0.5 より下なら、十分長く当てれば「0」に戻る
    erase = Rates(k_ab=0.0087, k_ba=0.039)
    _, _, ends = simulate_protocol(0.9, [Segment("erase", 200.0, erase)])
    assert ends[-1] == pytest.approx(erase.p_b_equilibrium, abs=1e-3)
    assert ends[-1] < 0.5

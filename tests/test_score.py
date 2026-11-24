"""スコア関数の検算テスト（紙で答えが分かるケース）。"""

import pytest

from chemcomp.score import geometric_mean, linear_up, log_window, score_switch


def test_linear_up_midpoint_and_clipping():
    assert linear_up(65.0, 30.0, 100.0) == pytest.approx(0.5)  # 真ん中なら 0.5
    assert linear_up(10.0, 30.0, 100.0) == 0.0  # 最低より下は 0
    assert linear_up(200.0, 30.0, 100.0) == 1.0  # 理想より上は 1


def test_log_window_shape():
    # 1 s〜60 s の上り坂の、対数軸での真ん中 √60 ≈ 7.75 s で 0.5
    assert log_window(60.0**0.5, 1.0, 60.0, 600.0, 3600.0) == pytest.approx(0.5)
    assert log_window(300.0, 1.0, 60.0, 600.0, 3600.0) == 1.0  # 理想の範囲
    assert log_window(0.5, 1.0, 60.0, 600.0, 3600.0) == 0.0  # 短すぎ
    assert log_window(1e5, 1.0, 60.0, 600.0, 3600.0) == 0.0  # 長すぎ


def test_geometric_mean_zero_kills():
    # 普通の平均なら 0.67 だが、1つでも 0 があれば 0
    assert geometric_mean([1.0, 1.0, 0.0]) == 0.0
    assert geometric_mean([0.25, 1.0, 1.0, 1.0]) == pytest.approx(0.25**0.25)


def test_score_switch_reports_failures():
    # 保持時間が1年（長すぎて用途Aには不向き）
    result = score_switch(t_half_s=3.2e7, p_pss=0.95, delta_lambda_nm=120.0)
    assert result.total == 0.0
    assert result.failed == ["t_half"]


def test_score_switch_perfect():
    result = score_switch(t_half_s=300.0, p_pss=0.95, delta_lambda_nm=120.0)
    assert result.total == pytest.approx(1.0)


# ---------- 第2版 ----------

from chemcomp.score import log_down, log_up, readout_on_off  # noqa: E402


def test_log_up_and_down_midpoints():
    # 100 と 10000 の対数軸での真ん中は 1000
    assert log_up(1e3, 1e2, 1e4) == pytest.approx(0.5)
    # 1 s と 10 s の対数軸での真ん中は √10 ≈ 3.16 s
    assert log_down(10**0.5, 1.0, 10.0) == pytest.approx(0.5)
    assert log_down(0.1, 1.0, 10.0) == 1.0


def test_readout_on_off_limited_by_incomplete_writing():
    # 完全に書き込めれば ON/OFF 比 = ε_B/ε_A そのもの
    assert readout_on_off(1.0, 10.0) == pytest.approx(10.0)
    # 書き込み率 90% だと、比は 0.9·r + 0.1 に下がる
    assert readout_on_off(0.9, 10.0) == pytest.approx(9.1)
    # 書き込み 50% では大きく下がる
    assert readout_on_off(0.5, 10.0) == pytest.approx(5.5)


def test_missing_values_are_reported_not_scored():
    r = score_switch(t_half_s=300.0, p_pss=0.95, delta_lambda_nm=120.0, on_off=20.0)
    assert r.missing == ["write_time", "cycles"]
    assert set(r.parts) == {"t_half", "p_pss", "delta_lambda", "on_off"}
    assert r.total == pytest.approx(1.0)


def test_fatigue_failure_kills_score():
    r = score_switch(300.0, 0.95, 120.0, on_off=20.0, write_time_s=0.5, cycles=50.0)
    assert r.total == 0.0
    assert r.failed == ["cycles"]

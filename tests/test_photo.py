"""光反応の速度定数の検算テスト。"""

import pytest

from chemcomp.photo import absorption_cross_section_cm2, photo_rate_per_s, photon_flux_per_cm2_s


def test_cross_section_conversion():
    # 教科書の換算式 σ [cm²] ≈ 3.82e-21 × ε [M⁻¹cm⁻¹]
    assert absorption_cross_section_cm2(1.0) == pytest.approx(3.82e-21, rel=1e-3)


def test_photon_flux():
    # 365 nm の光子1個 ≈ 5.44e-19 J なので、1 mW/cm² ≈ 1.84e15 photons/(cm²·s)
    assert photon_flux_per_cm2_s(1.0, 365.0) == pytest.approx(1.84e15, rel=1e-2)


def test_rate_is_linear_in_each_factor():
    base = photo_rate_per_s(1e4, 0.1, 10.0, 365.0)
    assert photo_rate_per_s(2e4, 0.1, 10.0, 365.0) == pytest.approx(2 * base)  # ε を2倍
    assert photo_rate_per_s(1e4, 0.2, 10.0, 365.0) == pytest.approx(2 * base)  # Φ を2倍
    assert photo_rate_per_s(1e4, 0.1, 20.0, 365.0) == pytest.approx(2 * base)  # 光を2倍

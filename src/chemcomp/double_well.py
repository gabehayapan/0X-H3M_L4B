"""2状態分子スイッチの1次元二重井戸モデルと熱緩和寿命。

ポテンシャル（反応座標 x は無次元。x≈-1 が状態A、x≈+1 が状態B）:

    V(x) = E_b * (x^2 - 1)^2 + delta * x / 2      [eV]

- E_b  : 対称な場合 (delta=0) の障壁高さ [eV]
- delta: 2状態のエネルギー差のおおよその大きさ [eV]（正なら状態Bが不安定 = 準安定状態）

熱緩和（準安定状態 B → 安定状態 A）の速度はアイリング式で見積もる:

    k   = (k_B T / h) * exp(-ΔE‡ / k_B T)     [1/s]
    t½  = ln 2 / k                            [s]

ΔE‡ は「準安定な井戸の底」から「障壁の頂上」までのエネルギー差。
注: ここでは自由エネルギー ΔG‡ をポテンシャルエネルギー ΔE‡ で近似している（エントロピー項は無視）。
"""

import numpy as np

from chemcomp.constants import H_EV_S, K_B_EV


def potential(x: np.ndarray, barrier_ev: float, asymmetry_ev: float = 0.0) -> np.ndarray:
    """二重井戸ポテンシャル V(x) [eV] を返す。"""
    return barrier_ev * (x**2 - 1.0) ** 2 + 0.5 * asymmetry_ev * x


def stationary_points(barrier_ev: float, asymmetry_ev: float = 0.0) -> dict[str, float]:
    """V'(x) = 0 の3つの根（左の極小・頂上・右の極小）の x 座標を返す。

    V'(x) = 4 E_b x^3 - 4 E_b x + delta/2 = 0 は3次方程式なので np.roots で解析的に解く。
    """
    coeffs = [4.0 * barrier_ev, 0.0, -4.0 * barrier_ev, 0.5 * asymmetry_ev]
    roots = np.roots(coeffs)
    real = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
    if real.size != 3:
        raise ValueError("非対称性が大きすぎて二重井戸になっていない (極小が1つしかない)")
    left, top, right = real
    return {"min_A": left, "barrier_top": top, "min_B": right}


def reverse_barrier_ev(barrier_ev: float, asymmetry_ev: float = 0.0) -> float:
    """準安定状態 B (右の井戸) から頂上までの障壁 ΔE‡ [eV]。"""
    pts = stationary_points(barrier_ev, asymmetry_ev)
    v_top = potential(pts["barrier_top"], barrier_ev, asymmetry_ev)
    v_b = potential(pts["min_B"], barrier_ev, asymmetry_ev)
    return float(v_top - v_b)


def forward_barrier_ev(barrier_ev: float, asymmetry_ev: float = 0.0) -> float:
    """安定状態 A (左の井戸) から頂上までの障壁 [eV]。A→B の熱的な乗り越えに対応。"""
    pts = stationary_points(barrier_ev, asymmetry_ev)
    v_top = potential(pts["barrier_top"], barrier_ev, asymmetry_ev)
    v_a = potential(pts["min_A"], barrier_ev, asymmetry_ev)
    return float(v_top - v_a)


def eyring_rate(delta_e_ev: np.ndarray | float, temperature_k: float) -> np.ndarray | float:
    """アイリング式による速度定数 k [1/s]。"""
    kt = K_B_EV * temperature_k  # 熱エネルギー k_B T [eV]（298 K で約 0.0257 eV）
    prefactor = kt / H_EV_S  # 頻度因子 k_B T / h [1/s]（298 K で約 6.2e12 1/s）
    return prefactor * np.exp(-np.asarray(delta_e_ev) / kt)


def half_life_s(delta_e_ev: np.ndarray | float, temperature_k: float) -> np.ndarray | float:
    """熱緩和の半減期 t½ = ln2 / k [s]。これが分子メモリの「保持時間」に相当する。"""
    return np.log(2.0) / eyring_rate(delta_e_ev, temperature_k)

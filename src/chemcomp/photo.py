"""光で分子が切り替わる速さ（光反応の速度定数）を、分子の性質と光の強さから計算する。

    k_ph = σ × Φ × F                                       [1/s]

- σ : 吸収断面積 [cm²]。分子1個が光子を受け止める「的の大きさ」
      σ = ln(10) × 1000 × ε / N_A                         （ε はモル吸光係数 [M⁻¹cm⁻¹]）
      1000 は L → cm³ の換算、ln(10) は常用対数（吸光度）→ 自然対数の換算
- Φ : 量子収率 [−]。光子を1個吸収したときに形が変わる確率
- F : 光子フラックス [photons/(cm²·s)]。1秒間に 1 cm² を通る光子の数
      F = 光の強さ I [W/cm²] ÷ 光子1個のエネルギー hc/λ [J]

注: 試料が薄い（光が奥まで均一に届く）場合の近似。濃い試料では奥ほど光が弱くなる。
"""

import numpy as np

from chemcomp.constants import C_M_S, H_J_S, N_A_PER_MOL

ML_PER_L = 1000.0  # 1 L = 1000 cm³ [cm³/L]
MW_PER_W = 1000.0  # [mW/W]
NM_PER_M = 1e9  # [nm/m]


def absorption_cross_section_cm2(epsilon_per_m_cm: float) -> float:
    """モル吸光係数 ε [M⁻¹cm⁻¹] → 吸収断面積 σ [cm²]。"""
    return float(np.log(10.0) * ML_PER_L * epsilon_per_m_cm / N_A_PER_MOL)


def photon_flux_per_cm2_s(intensity_mw_cm2: float, wavelength_nm: float) -> float:
    """光の強さ [mW/cm²] と波長 [nm] → 光子フラックス [photons/(cm²·s)]。"""
    photon_energy_j = H_J_S * C_M_S / (wavelength_nm / NM_PER_M)  # 光子1個のエネルギー [J]
    return (intensity_mw_cm2 / MW_PER_W) / photon_energy_j


def photo_rate_per_s(epsilon_per_m_cm: float, quantum_yield: float,
                     intensity_mw_cm2: float, wavelength_nm: float) -> float:
    """光反応の速度定数 k_ph [1/s]。"""
    sigma = absorption_cross_section_cm2(epsilon_per_m_cm)
    flux = photon_flux_per_cm2_s(intensity_mw_cm2, wavelength_nm)
    return sigma * quantum_yield * flux

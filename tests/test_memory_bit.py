"""1ビット化学メモリ部品の検算テスト（ARCHITECT.md §8.7）。"""

import importlib.util
from pathlib import Path

import pytest

from chemcomp.components.memory_bit import (
    LightPulse, MemoryBit, ReadoutConfig, SwitchParams, dark, read_bit, simulate_bit,
)
from chemcomp.constants import ROOM_TEMPERATURE_K
from chemcomp.double_well import eyring_rate, forward_barrier_ev, reverse_barrier_ev
from chemcomp.kinetics import Segment, simulate_protocol
from chemcomp.photo import absorption_cross_section_cm2, photon_flux_per_cm2_s

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m1_05_write_erase_cycle.py"


def load_m1_05():
    """M1-05 のスクリプトをモジュールとして読み込む（パラメータと結果を比べるため）。"""
    spec = importlib.util.spec_from_file_location("m1_05", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def m1_05_params(m) -> SwitchParams:
    return SwitchParams(
        barrier_ev=m.BARRIER_EV, asymmetry_ev=m.ASYMMETRY_EV, phi_ab=m.PHI_AB, phi_ba=m.PHI_BA,
        epsilon_a={m.WRITE_NM: m.EPS_A_WRITE, m.ERASE_NM: m.EPS_A_ERASE},
        epsilon_b={m.WRITE_NM: m.EPS_B_WRITE, m.ERASE_NM: m.EPS_B_ERASE},
    )


def m1_05_pulses(m) -> list[LightPulse]:
    light = {"uv": (m.WRITE_NM, m.WRITE_MW_CM2), "vis": (m.ERASE_NM, m.ERASE_MW_CM2)}
    pulses = []
    for name, dur, kind in m.PROTOCOL:
        if kind == "dark":
            pulses.append(dark(dur, name))
        else:
            wavelength, intensity = light[kind]
            pulses.append(LightPulse(wavelength, intensity, dur, name))
    return pulses


# 1. M1-05 と同じ条件で同じ結果になる
def test_matches_m1_05_script():
    m = load_m1_05()
    rates = m.build_rates(ROOM_TEMPERATURE_K)
    _, _, expected = simulate_protocol(0.0, [Segment(n, d, rates[k]) for n, d, k in m.PROTOCOL])
    trace = simulate_bit(m1_05_params(m), m1_05_pulses(m), ROOM_TEMPERATURE_K)
    assert trace.pulse_end_p_b == pytest.approx(expected, abs=1e-9)


# 2. I-02b と同じ光の速さ（k_AB = 1.0 /s, k_BA = 0.1 /s）なら、光定常状態 ≈ 0.909
def test_reproduces_i02b_photostationary_state():
    lam, inten, phi = 365.0, 10.0, 0.2  # 波長 [nm], 強さ [mW/cm²], 量子収率 [−]
    rate_per_eps = absorption_cross_section_cm2(1.0) * phi * photon_flux_per_cm2_s(inten, lam)  # ε=1 での k [1/s]
    params = SwitchParams(barrier_ev=1.2, asymmetry_ev=0.5, phi_ab=phi, phi_ba=phi,
                          epsilon_a={lam: 1.0 / rate_per_eps}, epsilon_b={lam: 0.1 / rate_per_eps})
    trace = simulate_bit(params, [LightPulse(lam, inten, 100.0)], ROOM_TEMPERATURE_K)
    # 熱の速さ（B→A ≈ 3.7e-4 /s）も足されるので、期待値は 1/1.1 から少しずれる
    k_th_ab = float(eyring_rate(forward_barrier_ev(1.2, 0.5), ROOM_TEMPERATURE_K))
    k_th_ba = float(eyring_rate(reverse_barrier_ev(1.2, 0.5), ROOM_TEMPERATURE_K))
    expected = (1.0 + k_th_ab) / (1.0 + k_th_ab + 0.1 + k_th_ba)
    assert trace.pulse_end_p_b[-1] == pytest.approx(expected, abs=1e-6)
    assert trace.pulse_end_p_b[-1] == pytest.approx(1.0 / 1.1, abs=1e-3)  # I-02b の 0.909


# 3. apply() を分けて呼んでも、まとめて呼んでも同じ
def test_apply_split_equals_apply_once():
    m = load_m1_05()
    params, pulses = m1_05_params(m), m1_05_pulses(m)
    readout = ReadoutConfig(wavelength_nm=m.ERASE_NM)
    once = MemoryBit(params, ROOM_TEMPERATURE_K, readout)
    once.apply(pulses)
    split = MemoryBit(params, ROOM_TEMPERATURE_K, readout)
    split.apply(pulses[:3])
    split.apply(pulses[3:])
    assert split.p_b == pytest.approx(once.p_b, abs=1e-9)


def test_read_after_write_and_erase():
    m = load_m1_05()
    bit = MemoryBit(m1_05_params(m), ROOM_TEMPERATURE_K, ReadoutConfig(wavelength_nm=m.ERASE_NM))
    pulses = m1_05_pulses(m)
    bit.apply(pulses[:2])  # 書き込み → 保持
    assert bit.read().bit == 1
    bit.apply(pulses[2:4])  # 消去 → 保持
    assert bit.read().bit == 0


def test_read_signal_formula():
    # signal = (p·ε_B + (1−p)·ε_A) / ε_A。p=0 なら 1、p=1 なら ε_B/ε_A
    params = SwitchParams(1.2, 0.5, 0.2, 0.3, epsilon_a={450.0: 500.0}, epsilon_b={450.0: 1500.0})
    readout = ReadoutConfig(wavelength_nm=450.0)
    assert read_bit(params, 0.0, readout).signal == pytest.approx(1.0)
    assert read_bit(params, 1.0, readout).signal == pytest.approx(3.0)
    assert read_bit(params, 0.5, readout).signal == pytest.approx(2.0)


# 4. 入力チェック（ARCHITECT.md §8.6）
def test_input_validation():
    params = SwitchParams(1.2, 0.5, 0.2, 0.3, epsilon_a={365.0: 2.5e4, 450.0: 0.0},
                          epsilon_b={365.0: 2.0e3, 450.0: 1.5e3})
    with pytest.raises(ValueError):
        simulate_bit(params, [dark(1.0)], ROOM_TEMPERATURE_K, p_b0=1.5)
    with pytest.raises(ValueError):
        LightPulse(365.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        LightPulse(365.0, 1.0, 0.0)
    with pytest.raises(KeyError):  # 辞書にない波長の光
        simulate_bit(params, [LightPulse(532.0, 1.0, 1.0)], ROOM_TEMPERATURE_K)
    with pytest.raises(ValueError):  # 読み取り波長で ε_A = 0
        read_bit(params, 0.5, ReadoutConfig(wavelength_nm=450.0))
    # 暗所なら辞書に波長がなくてもよい（熱で A→B がごくわずか進むだけ）
    assert simulate_bit(params, [dark(1.0)], ROOM_TEMPERATURE_K).pulse_end_p_b[-1] == pytest.approx(0.0, abs=1e-9)

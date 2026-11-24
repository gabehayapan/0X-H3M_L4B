"""分子スイッチの「ゲート適性スコア」第2版（用途A：一時的な記憶、6指標）。

定義の詳細と根拠は RESEARCH_NOTES.md「I-03」を参照。

    1. 各指標を 0〜1 の満足度 d に変換する
    2. 総合スコア S = (d_1 × d_2 × … × d_n)^(1/n)   （幾何平均）
    3. 値が分からない指標（None）は「不明」として計算から外し、一覧で報告する
"""

from dataclasses import dataclass, field

import numpy as np


# ---------- 満足度 d への変換関数 ----------

def linear_up(x: float, low: float, ideal: float) -> float:
    """大きいほど良い指標（直線）。x ≤ low で 0、x ≥ ideal で 1。"""
    return float(np.clip((x - low) / (ideal - low), 0.0, 1.0))


def log_up(x: float, low: float, ideal: float) -> float:
    """大きいほど良い指標（対数軸で直線）。桁で効く量（回数、比）に使う。"""
    if x <= 0:
        return 0.0
    return float(np.clip((np.log10(x) - np.log10(low)) / (np.log10(ideal) - np.log10(low)), 0.0, 1.0))


def log_down(x: float, ideal: float, high: float) -> float:
    """小さいほど良い指標（対数軸で直線）。x ≤ ideal で 1、x ≥ high で 0。"""
    if x <= 0:
        return 1.0
    return float(np.clip((np.log10(high) - np.log10(x)) / (np.log10(high) - np.log10(ideal)), 0.0, 1.0))


def log_window(x: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    """対数軸上の台形。短すぎても長すぎてもダメな量（保持時間）の満足度。"""
    if x <= 0:
        return 0.0
    u = np.log10(x)
    up = (u - np.log10(low)) / (np.log10(ideal_low) - np.log10(low))
    down = (np.log10(high) - u) / (np.log10(high) - np.log10(ideal_high))
    return float(np.clip(min(up, down), 0.0, 1.0))


def geometric_mean(values: list[float]) -> float:
    """幾何平均。1つでも 0 があれば 0 になる。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(arr <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


# ---------- 物理量からの換算 ----------

def readout_on_off(p_pss: float, epsilon_ratio_read: float) -> float:
    """読み取り波長での 1/0 の信号比（ON/OFF比）[−]。

    0 の状態  : 全分子が形A   → 信号 ∝ ε_A
    1 の状態  : 割合 p_PSS が形B → 信号 ∝ p_PSS·ε_B + (1 − p_PSS)·ε_A
    epsilon_ratio_read = ε_B / ε_A（読み取り波長での値）

    書き込みが不完全（p_PSS < 1）だと、形Bがどれだけ強く吸収しても比は頭打ちになる。
    """
    s0 = 1.0
    s1 = p_pss * epsilon_ratio_read + (1.0 - p_pss)
    return max(s0, s1) / min(s0, s1)


# ---------- 用途Aの基準値（仮置き） ----------

@dataclass(frozen=True)
class UseCaseA:
    """用途A（計算の途中結果を一時的に覚え、待てば勝手に消える）の基準値。"""

    # 1. 保持時間 t½ [s]（対数の台形）
    t_half_low_s: float = 1.0  # これより短いと読む前に消える
    t_half_ideal_low_s: float = 60.0  # 理想範囲の下限（1分）
    t_half_ideal_high_s: float = 600.0  # 理想範囲の上限（10分）
    t_half_high_s: float = 3600.0  # これより長いと待っても消えない（1時間）
    # 2. 書き込み率 p_PSS [−]（直線）
    p_pss_low: float = 0.6  # 読み取りの境目 0.5 に余裕がない
    p_pss_ideal: float = 0.9
    # 3. 波長の分離 Δλ [nm]（直線）
    delta_lambda_low_nm: float = 30.0  # 2つの形を別の光で操作できない
    delta_lambda_ideal_nm: float = 100.0
    # 4. 読み取り ON/OFF 比 [−]（対数）
    on_off_low: float = 2.0  # 2倍未満の差では雑音と区別しにくい
    on_off_ideal: float = 10.0
    # 5. 書き込み時間 τ_w = 1/(k_AB + k_BA) [s]（対数、小さいほど良い）
    write_time_ideal_s: float = 1.0  # 保持時間（理想 ≥ 60 s）より十分短い
    write_time_high_s: float = 10.0  # 保持時間の下限付近まで遅いと使えない
    # 6. 疲労耐性 N_cyc [回]（対数）
    cycles_low: float = 1e2  # 1分ごとに書き換えると2時間足らずで寿命
    cycles_ideal: float = 1e4  # 1分ごとで約1週間


@dataclass
class ScoreResult:
    total: float  # 総合スコア S [0〜1]（分かっている指標だけで計算）
    parts: dict[str, float]  # 指標ごとの満足度 d
    failed: list[str] = field(default_factory=list)  # 足切り（d = 0）になった指標
    missing: list[str] = field(default_factory=list)  # 値が不明で計算から外した指標


def score_switch(t_half_s: float, p_pss: float, delta_lambda_nm: float,
                 on_off: float | None = None, write_time_s: float | None = None,
                 cycles: float | None = None, criteria: UseCaseA = UseCaseA()) -> ScoreResult:
    """6指標から総合スコアを計算する。4〜6 は None（不明）でもよい。"""
    c = criteria
    parts = {
        "t_half": log_window(t_half_s, c.t_half_low_s, c.t_half_ideal_low_s,
                             c.t_half_ideal_high_s, c.t_half_high_s),
        "p_pss": linear_up(p_pss, c.p_pss_low, c.p_pss_ideal),
        "delta_lambda": linear_up(delta_lambda_nm, c.delta_lambda_low_nm, c.delta_lambda_ideal_nm),
    }
    optional = {
        "on_off": (on_off, lambda v: log_up(v, c.on_off_low, c.on_off_ideal)),
        "write_time": (write_time_s, lambda v: log_down(v, c.write_time_ideal_s, c.write_time_high_s)),
        "cycles": (cycles, lambda v: log_up(v, c.cycles_low, c.cycles_ideal)),
    }
    missing = []
    for name, (value, to_d) in optional.items():
        if value is None:
            missing.append(name)
        else:
            parts[name] = to_d(value)

    failed = [name for name, d in parts.items() if d == 0.0]
    return ScoreResult(total=geometric_mean(list(parts.values())), parts=parts,
                       failed=failed, missing=missing)

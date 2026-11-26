"""2状態スイッチの集団（たくさんの分子）の時間変化：光で書き込み、熱で消える。

1分子ではなく、分子の集団のうち形Bにいる割合 p_B(t) を追う（p_A = 1 - p_B）。

    dp_B/dt = k_AB * p_A  -  k_BA * p_B                                   ... (1)

    k_AB = k_th_AB + k_ph_AB   （A→B：熱 ＋ 光）  [1/s]
    k_BA = k_th_BA + k_ph_BA   （B→A：熱 ＋ 光）  [1/s]

- 熱による速度 k_th は I-02 の障壁からアイリング式で計算する
- 光による速度 k_ph は光が当たっている間だけ働く（ここでは値を直接与える）
  本来は k_ph = (吸収断面積 σ) × (量子収率 Φ) × (光子フラックス) で決まる

式(1)は線形なので解析解がある。速度が一定の区間では
    p_B(t) = p_eq + (p_B(0) - p_eq) * exp(-(k_AB + k_BA) * t),   p_eq = k_AB / (k_AB + k_BA)
テストではこの解析解と数値解 (scipy.integrate.solve_ivp) を比べる。
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Rates:
    """速度定数の組 [1/s]。"""

    k_ab: float  # A→B の合計速度 [1/s]
    k_ba: float  # B→A の合計速度 [1/s]

    @property
    def p_b_equilibrium(self) -> float:
        """十分長い時間後に落ち着く p_B（光照射中なら「光定常状態」）。"""
        return self.k_ab / (self.k_ab + self.k_ba)


def rhs(_t: float, y: np.ndarray, rates: Rates) -> np.ndarray:
    """式(1)の右辺。y = [p_B]。"""
    p_b = y[0]
    return np.array([rates.k_ab * (1.0 - p_b) - rates.k_ba * p_b])


def simulate_segment(p_b0: float, rates: Rates, t_eval: np.ndarray) -> np.ndarray:
    """速度が一定の1区間を数値積分し、t_eval 時点での p_B を返す（t_eval[0] = 0 とする）。

    LSODA は硬い(stiff)系と硬くない系を自動で切り替えるソルバ。
    この系は熱(1e-4 1/s)と光(1 1/s)で時間スケールが4桁違うので、安全側で選んでいる。
    """
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [p_b0], t_eval=t_eval, args=(rates,),
                    method="LSODA", rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def analytic_segment(p_b0: float, rates: Rates, t: np.ndarray) -> np.ndarray:
    """式(1)の解析解（速度が一定の区間）。"""
    p_eq = rates.p_b_equilibrium
    return p_eq + (p_b0 - p_eq) * np.exp(-(rates.k_ab + rates.k_ba) * t)


@dataclass(frozen=True)
class Segment:
    """操作の1区間（この間は速度定数が一定）。"""

    label: str  # 区間の名前（例: "write", "hold", "erase"）
    duration_s: float  # 区間の長さ [s]
    rates: Rates  # この区間の速度定数（光の有無・色で変わる）


def simulate_protocol(p_b0: float, segments: list[Segment],
                      points_per_segment: int = 200) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """区間を順番につないでシミュレートする（前の区間の最後の p_B が次の区間の初期値）。

    戻り値: (時刻 [s], p_B, 各区間の終わりの p_B のリスト)
    """
    times, values, ends = [np.array([0.0])], [np.array([p_b0])], []
    t0, p = 0.0, p_b0
    for seg in segments:
        t_local = np.linspace(0.0, seg.duration_s, points_per_segment)
        p_seg = simulate_segment(p, seg.rates, t_local)
        times.append(t0 + t_local[1:])
        values.append(p_seg[1:])
        t0, p = t0 + seg.duration_s, float(p_seg[-1])
        ends.append(p)
    return np.concatenate(times), np.concatenate(values), ends


def time_to_threshold(p_b_start: float, rates_dark: Rates, threshold: float) -> float:
    """暗所で p_B が threshold を下回るまでの時間 [s]（「1」と読めなくなるまでの時間）。

    解析解を t について解いた: t = ln((p0 - p_eq) / (th - p_eq)) / (k_AB + k_BA)
    """
    p_eq = rates_dark.p_b_equilibrium
    if not (p_eq < threshold < p_b_start):
        raise ValueError("threshold は p_eq と p_b_start の間にある必要がある")
    return float(np.log((p_b_start - p_eq) / (threshold - p_eq)) / (rates_dark.k_ab + rates_dark.k_ba))

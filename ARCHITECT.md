# ARCHITECT.md - 技術スタックとアーキテクチャ

> ステータス: **提案（未合意）**。AGENTS.md §2 に従い、ここに書いたライブラリは**合意してからフェーズごとに導入する**。最初から全部入れることはしない。
> 最終更新: 2026-11-15

---

## 1. 結論（推奨）

| 項目 | 推奨 | 一言理由 |
|---|---|---|
| 言語 | **Python 3.12** | 化学・量子化学・ML・量子計算のライブラリがすべてPythonで揃う唯一の言語 |
| 環境・依存管理 | **uv** + `pyproject.toml` + `uv.lock` | 高速、ロックファイルで再現性を担保できる |
| 数値計算の基盤 | **NumPy / SciPy / Matplotlib** | AGENTS.md §5 の「依存の少ないプロトタイプ」方針に合う。式が透けて見える |
| テスト | **pytest** | 物理的な検算（保存則、既知の解析解）をテストとして残す |
| 実行形態 | スクリプト（`scripts/`）＋ 再利用部分はパッケージ（`src/chemcomp/`） | ノートブックは探索用に限定し、結論はスクリプトで再現可能にする |

速度が問題になった場合は、まず計測用のIssueを立てる（AGENTS.md §5）。その後の手段は Numba → JAX → C++/Rust 拡張 の順に検討する。

---

## 2. 言語の比較

| 言語 | 長所 | 短所 | 判断 |
|---|---|---|---|
| **Python** | RDKit, PySCF, Qiskit/PennyLane, PyTorch, SciPy が揃う。学習資料が最多 | 純Pythonのループは遅い（NumPyのベクトル化で回避） | **主言語** |
| Julia | DifferentialEquations.jl が強力。ODE/PDEが速い | 化学情報学（RDKit相当）と量子化学のエコシステムが薄い。2言語管理のコスト | 不採用。反応拡散の大規模計算が必要になったら再検討 |
| C++ / Rust | 最速。大規模格子シミュレーション向き | プロトタイプが遅く、ブラックボックス化しやすい | ボトルネックが計測で判明した後に、拡張モジュールとして限定的に |
| MATLAB | ODEソルバが使いやすい | 有償、化学DBやMLとの連携が弱い | 不採用 |

---

## 3. フェーズ別の技術スタック（段階導入）

各ライブラリは**そのフェーズのIssueで必要になった時点で**合意してから追加する。

| Phase | 用途 | 最初に使うもの | 後で検討する候補 | 備考 |
|---|---|---|---|---|
| 1 原理定義 | ポテンシャル曲線、速度論 | NumPy, SciPy, Matplotlib | - | I-02 はこれだけで完結 |
| 2 DB・権利調査 | ドキュメント | Markdown | - | コード不要 |
| 3 データ取得 | API取得・前処理・キャッシュ | `urllib`/`requests`, pandas, **RDKit**（SMILES処理） | `mp-api`（Materials Project公式クライアント）, PubChemPy | 生データは `data/raw/` にキャッシュし、出典メタデータを併記 |
| 4 探索（古典） | 物性予測、ベイズ最適化 | scikit-learn（ガウス過程） | PyTorch + PyTorch Geometric（GNN）, BoTorch | まずGPで小規模に。GNNはデータ量が十分なときのみ |
| 4 探索（量子） | VQE等 | **NumPy で行列から自作**（I-09） | PySCF（ハミルトニアン生成）, Qiskit または PennyLane | 量子優位は期待しない（PROJECT.md C5）。学習目的と明記 |
| 4/5 量子化学 | 構造最適化、励起状態 | - | PySCF（DFT/TDDFT）, xtb（半経験的、高速） | 光スイッチは励起状態が本質（C3） |
| 5 動作検証 | 反応ネットワークODE、反応拡散PDE | `scipy.integrate.solve_ivp`, NumPy差分法 | Numba（高速化）, JAX（自動微分＋高速化） | 硬い(stiff)系は `method="LSODA"`/`"BDF"` |
| 6 最適化・生成 | 分子生成、強化学習 | - | SELFIES, REINVENT系, BoTorch | Phase 4 の成果に依存。時期が来たら再設計 |

### トレードオフの要点
- **Qiskit vs PennyLane**: Qiskit はIBM実機との接続とエコシステムが強い。PennyLane は自動微分と PyTorch/JAX との統合が強い。Phase 4 の時点で、どちらにするか合意する。
- **PySCF vs xtb**: PySCF は理論が明示的で学習向きだが遅い。xtb は高速だが近似が粗い。精度が必要な少数分子には PySCF、スクリーニングには xtb と使い分ける想定。
- **RDKit**: 事実上の標準。ただしC++拡張を含む重い依存なので、Phase 3 で必要になってから入れる。

---

## 4. ディレクトリ構成（案）

```
0X-H3M_L4B/
├── AGENTS.md / PROJECT.md / ISSUES.md / ARCHITECT.md / RESEARCH_NOTES.md
├── pyproject.toml          # 依存はフェーズごとに追加（uvで管理）
├── uv.lock
├── src/chemcomp/
│   ├── constants.py        # 物理定数と単位換算（全て単位コメント付き）
│   ├── models/             # 二重井戸、反応ネットワーク、反応拡散などの物理モデル
│   ├── data/               # DB取得・キャッシュ・出典メタデータ
│   ├── search/             # GP / ベイズ最適化 / VQE
│   └── sim/                # ODE/PDE ソルバのラッパー
├── scripts/                # Issueごとの実行スクリプト（例: i02_double_well.py）
├── tests/                  # 物理的な検算テスト
├── data/
│   ├── raw/                # 取得データのキャッシュ（.gitignore、出典JSONのみコミット）
│   └── processed/
├── outputs/figures/        # 生成した図（再生成可能なものは .gitignore）
└── notebooks/              # 探索用（結論はスクリプトに移す）
```

最初は `scripts/` 単体のスクリプトで始める。2つ以上のIssueで同じコードが必要になった時点で `src/chemcomp/` に切り出す（早すぎる抽象化を避ける）。

---

## 5. データパイプライン（Phase 3 以降の想定）

```
[外部DB API] → fetch（レート制限・リトライ）
             → data/raw/<source>/<id>.json  + source_meta.json（URL, 取得日時, ライセンス, バージョン）
             → 前処理（単位をeVに統一、SMILES正規化）
             → data/processed/*.parquet
             → 探索 / シミュレーション
```

- **キャッシュ優先**: ローカルにあればAPIを叩かない
- **出典の保持**: すべての派生データが `source_meta.json` を辿れるようにする
- **単位の統一**: 内部では eV、Å、K、s を基本単位とし、変換は `constants.py` に集約する

---

## 6. 規約（AGENTS.md §5 の具体化）

- 乱数: `numpy.random.default_rng(seed)` を明示的に渡す（グローバルシードは使わない）
- 定数: `K_B_EV = 8.617333262e-5  # ボルツマン定数 [eV/K]` のように単位と意味を必ず併記
- フォーマット / Lint: ruff（合意後に導入）
- 図: 軸ラベルに単位を必ず入れる

---

## 7. 未決事項（ユーザーと合意が必要）

1. Python 3.12 + uv で始めてよいか
2. Phase 1 の I-02 で `pyproject.toml` を作り、NumPy / SciPy / Matplotlib / pytest の4つだけを入れてよいか
3. ノートブックをどこまで使うか（探索のみ、またはIssueの成果物にも認めるか）

---

## 8. 部品インターフェース：1ビット化学メモリ（M1-06）

> ステータス: **設計（M1-07 で実装する）**。2026-11-27
> 目的: I-02 / I-02b / M1-05 で作った「書き込み・保持・消去」を、回路（M2〜M3）やビジュアライザ（M4）から**同じ呼び方で使える部品**にする。

### 8.1 部品の見え方（ブラックボックスとして）

```
入力: 光パルスの列 ──▶ ┌──────────────────────────┐ ──▶ 出力: 読み取り結果
  (波長, 強さ, 時間)    │ 1ビット化学メモリ          │       (ビット 0/1, p_B, 信号の強さ)
                        │ 内部状態: p_B（形Bの割合） │
  条件: 温度 T          │ 中身: 分子のパラメータ     │
                        └──────────────────────────┘
```

- **入力**は「光パルスの列」だけ。書き込み・消去・保持（暗所）・読み取り光は、すべて光パルスの違い（波長・強さ・時間）として表す
- **内部状態**は p_B（0〜1）の1つだけ。I-02b 以降の2状態モデルと同じ
- **出力**は「今 p_B がいくつか」と「それを 0/1 としてどう読むか」

### 8.2 データ構造

| 名前 | 種類 | フィールド（単位） | 意味 |
|---|---|---|---|
| `LightPulse` | frozen dataclass | `wavelength_nm` [nm], `intensity_mw_cm2` [mW/cm²], `duration_s` [s] | 1つの操作区間。`intensity_mw_cm2 = 0` が暗所（保持） |
| `SwitchParams` | frozen dataclass | `barrier_ev` [eV], `asymmetry_ev` [eV], `phi_ab` [−], `phi_ba` [−], `epsilon_a` / `epsilon_b`: `dict[float, float]`（波長 [nm] → ε [M⁻¹cm⁻¹]） | 分子の性質。熱の速さは障壁から（I-02）、光の速さは ε・Φ から（I-03 第2版 `photo.py`）計算する |
| `ReadoutConfig` | frozen dataclass | `wavelength_nm` [nm], `threshold` [−]（既定 0.5） | 読み取り方。p_B ≥ threshold を「1」とする |
| `ReadResult` | dataclass | `bit` (0/1), `p_b` [−], `signal` [−] | 読み取り結果。`signal` は「0の状態（全分子が形A）の吸光度」を 1 とした相対値 |
| `Trace` | dataclass | `t` [s] (配列), `p_b` [−] (配列), `pulse_end_p_b` [−] (リスト) | 時間変化の記録。図やビジュアライザ（M4）に渡す |

吸光スペクトルは、最初は**使う波長だけの辞書**で持つ（例: `{365.0: 2.5e4, 450.0: 5.0e2}`）。

| 方式 | 長所 | 短所 | 判断 |
|---|---|---|---|
| **波長ごとの辞書（採用）** | 単純。文献の表の値をそのまま入れられる | 辞書にない波長は使えない | 第1版はこれ。辞書にない波長は `KeyError` にして、黙って間違えないようにする |
| スペクトル全体（ガウス関数の和など） | どの波長でも計算できる | パラメータが増え、根拠のない形を仮定しがち | M6（実在分子）で必要になったら |

### 8.3 関数とクラス

**中心は「状態を持たない関数」**、その上に「状態を持つ薄いクラス」を重ねる。

```python
# 純粋な関数（同じ入力なら必ず同じ結果。テストしやすい）
def rates_for_pulse(params: SwitchParams, pulse: LightPulse, temperature_k: float) -> Rates: ...
def simulate_bit(params, pulses: list[LightPulse], temperature_k: float, p_b0: float = 0.0) -> Trace: ...
def read_bit(params, p_b: float, readout: ReadoutConfig) -> ReadResult: ...

# 状態を持つクラス（回路やビジュアライザから1ステップずつ動かす用）
class MemoryBit:
    def __init__(self, params, temperature_k, readout, p_b0=0.0): ...
    p_b: float                                         # 今の状態
    def apply(self, pulses: list[LightPulse]) -> Trace  # 光を当てて状態を進める
    def read(self) -> ReadResult                        # 今の状態を読む（副作用なし）
```

| 方式 | 長所 | 短所 |
|---|---|---|
| 関数だけ | テストしやすい。結果が再現できる | 回路やゲームで「今の状態」を毎回渡す必要がある |
| クラスだけ | 使う側が楽 | 隠れた状態のせいでテストしにくい |
| **関数 ＋ 薄いクラス（採用）** | 計算の中身は関数でテストし、使うときはクラスで楽をする | ファイルが少し長くなる |

### 8.4 計算の中身（既存コードの組み合わせ）

各パルスについて速度定数を作り、M1-05 の `simulate_protocol` でつなぐだけ。

```
k_AB = k_th_AB(ΔE‡_AB, T) + k_ph(ε_A(λ), Φ_AB, I, λ)       ← double_well.eyring_rate ＋ photo.photo_rate_per_s
k_BA = k_th_BA(ΔE‡_BA, T) + k_ph(ε_B(λ), Φ_BA, I, λ)
暗所 (I = 0) なら k_ph = 0 なので、辞書に波長がなくてもよい
→ Segment(label, duration, Rates(k_AB, k_BA)) の列 → kinetics.simulate_protocol
```

読み取り信号（読み取り波長 λ_r で）:

```
signal = (p_B·ε_B(λ_r) + (1 − p_B)·ε_A(λ_r)) / ε_A(λ_r)      （I-03 第2版の ON/OFF 比と同じ考え方）
bit    = 1 if p_B ≥ threshold else 0
```

### 8.5 読み取りの扱い（M1-08 への布石）

- `read()` は**理想的な読み取り**（状態を変えない）として作る
- 実際の読み取りは「読み取り波長の弱い光を短時間当てる」ことなので、**副作用は `LightPulse` として `apply()` に入れれば表せる**。M1-08 ではこの仕組みで「1回の読み取りで何%書き換わるか」を計算する。インターフェースを変える必要はない

### 8.6 守るべき条件（入力チェック）

| 条件 | 違反したとき |
|---|---|
| `0 ≤ p_b0 ≤ 1` | `ValueError` |
| `intensity_mw_cm2 ≥ 0`、`duration_s > 0` | `ValueError` |
| 光を当てる（I > 0）波長が `epsilon_a` と `epsilon_b` の両方にある | `KeyError`（黙って ε = 0 にしない） |
| 読み取り波長での `epsilon_a > 0` | `ValueError`（信号を ε_A で割るため） |

### 8.7 置き場所とテスト（M1-07）

- コード: `src/chemcomp/components/memory_bit.py`（`components/` は回路部品をまとめる場所。M2 のゲートもここに置く）
- テスト `tests/test_memory_bit.py` で確かめること:
  1. M1-05 のスクリプトと同じパラメータ・同じ手順で、各区間の終わりの p_B が一致する
  2. I-02b と同じ光の速さ（k_ph,AB = 1.0 /s, k_ph,BA = 0.1 /s）になるよう ε と強さを選ぶと、I-02b と同じ光定常状態（≈ 0.909）になる
  3. `MemoryBit.apply()` を2回に分けて呼んでも、1回でまとめて呼んでも結果が同じ（状態が正しく引き継がれる）
  4. 8.6 の入力チェックがそれぞれエラーになる

### 8.8 今は入れないもの（後の Issue で検討）

| 機能 | いつ | 理由 |
|---|---|---|
| 疲労（劣化）のカウンタ | M4-07 / M4-15 | ゲームでの表示用。何を「1回」と数えるか（書き込み回数 / 切り替わった分子の総量）を決める必要がある |
| パルスごとの温度変化 | M4-15 | 温度スライダーで必要になってから |
| 濃い試料での光の減衰 | M6 | 今は「薄い試料（光が均一に届く）」近似（`photo.py` の注記と同じ） |
| 回路の他の部品への出力（光 → 濃度） | M2-14 | 部品どうしのつなぎ方は M2-01 の決定に従う |

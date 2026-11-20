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

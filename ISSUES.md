# ISSUES.md - バックログと進捗

> ルール（AGENTS.md §3）: `IN_PROGRESS` は常に1つだけ。1 Issue = 15〜30分で理解・検証できるサイズ。
> GitHub Issues と対応するものは `#番号` を併記する。候補一覧の背景は `PROJECT.md` §3〜4 を参照。

## IN_PROGRESS
- **I-02** [#2](https://github.com/gabehayapan/0X-H3M_L4B/issues/2) 二重井戸モデル — 最小プロダクトとして実装済み、ユーザーレビュー待ち
  - 実行: `uv run python scripts/i02_double_well.py` / テスト: `uv run pytest`
  - メモ: Issue本文の検算基準「ΔE‡≈1 eV で日〜年」は誤り。アイリング式では 1.0 eV ≈ 2.5 時間、1.1 eV ≈ 5日（アゾベンゼンcis体の半減期〜数日は ΔE‡≈1.1 eV に相当）
- **I-02b**（GitHub未起票）光による書き込みと熱による消失の時間発展（2状態速度論ODE）— 実装済み、ユーザーレビュー待ち
  - 実行: `uv run python scripts/i02b_write_and_decay.py`
  - 結果: 光定常状態は p_B ≈ 0.91（逆向きの光反応があるため100%は書き込めない）。25 °C で「1」と読める時間 ≈ 1,600 s（半減期 1,870 s より短い）、50 °C では ≈ 80 s

## TODO — Phase 1: 原理定義（GitHub起票済み）
| ID | GitHub | タイトル | 関連課題 | 依存 |
|---|---|---|---|---|
| I-01 | [#1](https://github.com/gabehayapan/0X-H3M_L4B/issues/1) | 4系統の比較と主軸アプローチの選定 | P1, P2, P6 | - |
| I-02 | [#2](https://github.com/gabehayapan/0X-H3M_L4B/issues/2) | 2状態分子スイッチの二重井戸モデルと熱緩和寿命の計算 | P4, C1 | - |
| I-03 | [#3](https://github.com/gabehayapan/0X-H3M_L4B/issues/3) | 分子論理ゲート適性スコアの数式定義 | C1 | I-01, I-02 |

## BACKLOG（未起票。Phase 1 完了後に見直す）
| ID | Phase | タイトル | 関連課題 |
|---|---|---|---|
| I-04 | 2 | QM9 / Materials Project / PubChem の公式ライセンス比較表 | C4 |
| I-05 | 2 | 光スイッチ専用データセットの調査 | C2 |
| I-06 | 3 | QM9から10分子のSMILESとHOMO-LUMOギャップを表示する最小スクリプト | - |
| I-07 | 3 | QM9の重原子数分布の可視化（光スイッチ骨格のカバー範囲確認） | C2 |
| I-08 | 4 | 光スイッチ λmax のガウス過程ベースライン | C2 |
| I-09 | 4 | H₂（2量子ビット）VQEをNumPyで実装し厳密対角化と比較 | C5 |
| I-10 | 5 | 反応ネットワークODEによるAND/ORゲートの定常応答 | C6 |
| I-11 | 5 | 2ゲート直列接続のカスケード性検証 | P1 |
| I-12 | 5 | ランダム反応ネットワークによるリザバー計算の再現 | P6 |
| I-13 | 5 | 1次元反応拡散モデルでの波の伝播・衝突シミュレーション | P2 |
| I-14 | 6 | ゲート適性スコアを目的関数とするベイズ最適化の最小例 | C1 |

## DONE
（なし）

<!-- 完了時の記録フォーマット:
### I-XX: タイトル (#N) — YYYY-MM-DD
- 結論:
- 得られた知見 / 次への示唆:
-->

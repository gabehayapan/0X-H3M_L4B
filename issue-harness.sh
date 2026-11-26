#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 設定
# ==========================================
ISSUES_FILE="${ISSUES_FILE:-./ISSUES.md}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"        # デフォルト3回までに制限
LOG_DIR="${LOG_DIR:-./.claude-harness-logs}"
COMPLETE_TOKEN="ALL_ISSUES_COMPLETED"
# 非対話実行時の権限設定（環境に合わせて調整してください）
CLAUDE_PERMISSION_FLAG="${CLAUDE_PERMISSION_FLAG:---permission-mode acceptEdits}"

# ==========================================
# 引数・事前チェック
# ==========================================
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MAIN_SESSION_ID> [ISSUES_FILE]"
  echo "Example: $0 12345678-abcd-efgh-ijkl-1234567890ab ./ISSUES.md"
  exit 1
fi

MAIN_SESSION_ID="$1"
if [[ $# -ge 2 ]]; then
  ISSUES_FILE="$2"
fi

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "Error: ${ISSUES_FILE} が見つかりません。"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: 作業ツリーに未コミットの変更があります。コミットまたは stash してから実行してください。"
  exit 1
fi

mkdir -p "$LOG_DIR"
SESSION_MAP_FILE="$LOG_DIR/forked_sessions.tsv"
touch "$SESSION_MAP_FILE"

# ==========================================
# プロンプト定義
# ==========================================
PROMPT=$(cat <<EOF
メインセッションで共有した設計方針・規約に従い、\`${ISSUES_FILE}\` に記載されている Issue を順番に処理します。

## 指示
1. \`${ISSUES_FILE}\` を読み込み、まだ完了していない最初の Issue を **1つだけ** 特定してください。
2. もし未完了の Issue が1つも残っていない場合は、ファイルを一切変更せず、出力の最後に必ず \`${COMPLETE_TOKEN}\` と出力して終了してください。
3. 未完了の Issue がある場合は、その1つだけを実装してください。
   - 必要なコードの実装とテストの追加を行うこと
   - 既存のテスト・リントが通ることを確認すること
   - \`${ISSUES_FILE}\` 内の該当 Issue に完了したことが分かる印（元のフォーマットに合わせて \`[x]\` や \`[DONE]\` など）を付けること
   - 実装と \`${ISSUES_FILE}\` の更新をまとめてコミットすること
EOF
)

# ==========================================
# メインループ: 毎回メインセッションからフォークして1件ずつ実行
# ==========================================
for ((i = 1; i <= MAX_ITERATIONS; i++)); do
  ITER_ID=$(printf "%03d" "$i")
  LOG_FILE="$LOG_DIR/run-${ITER_ID}.json"

  echo "============================================================"
  echo "[Run ${ITER_ID}/${MAX_ITERATIONS}] メインセッションからフォークして実行中..."
  echo "============================================================"

  if claude -p "$PROMPT" \
    --resume "$MAIN_SESSION_ID" \
    --fork-session \
    $CLAUDE_PERMISSION_FLAG \
    --output-format json \
    > "$LOG_FILE"; then

    FORKED_SESSION_ID=$(jq -r '.session_id // "unknown"' "$LOG_FILE")
    RESULT_TEXT=$(jq -r '.result // ""' "$LOG_FILE")

    # 全 Issue 完了シグナルの判定
    if grep -q "$COMPLETE_TOKEN" <<< "$RESULT_TEXT"; then
      echo -e "${ITER_ID}\t${FORKED_SESSION_ID}\tCOMPLETED" >> "$SESSION_MAP_FILE"
      echo "すべての Issue が完了しました。(Session: ${FORKED_SESSION_ID})"
      exit 0
    fi

    echo -e "${ITER_ID}\t${FORKED_SESSION_ID}\tSUCCESS" >> "$SESSION_MAP_FILE"
    echo "[Run ${ITER_ID}] 1件の Issue を処理しました。(Forked Session: ${FORKED_SESSION_ID})"
  else
    FORKED_SESSION_ID=$(jq -r '.session_id // "unknown"' "$LOG_FILE" 2>/dev/null || echo "unknown")
    echo -e "${ITER_ID}\t${FORKED_SESSION_ID}\tFAILED" >> "$SESSION_MAP_FILE"
    echo "[Run ${ITER_ID}] 失敗しました。ログを確認してください: ${LOG_FILE}"
    echo "個別デバッグ用コマンド: claude --resume ${FORKED_SESSION_ID}"
    exit 1
  fi
done

echo "上限回数 (${MAX_ITERATIONS}回) に達したため終了しました。"

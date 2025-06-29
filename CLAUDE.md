- YOU MUST: タスク完了またはユーザーにメッセージを返すときに最後に一度だけ `afplay /System/Library/Sounds/Funk.aiff -v 30 ` コマンドを実行して通知する

# ClaudeCode ログ可視化ツール「ccwatch」

## プロジェクト概要
ClaudeCodeのログファイル（~/.claude/projects/**/*.jsonl）を監視・可視化するWebアプリケーション。
DuckDBのJSONL直接読み込み機能を活用し、ゼロコピーで高速なクエリ処理を実現。

## 開発サイクル

### Streamlitアプリの起動
```bash
# 依存関係のインストール
uv sync

# アプリケーションの起動
uv run streamlit run src/app.py

# バックグラウンドで起動
uv run streamlit run src/app.py > streamlit.log 2>&1 &

# プロセスの停止
pkill -f "streamlit run src/app.py"
```

### DuckDBでのデータ構造確認
```bash
# ファイルのスキーマを確認
duckdb -c "DESCRIBE SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 1;"

# 実際のデータを確認
duckdb -c "SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 5;"
```

### エラー確認
- エラーは `error.log` に自動記録される
- `cat error.log` でエラーを確認

## コード品質管理（Ruff）

### 必須実行コマンド
```bash
# Lintチェックと自動修正
uv run ruff check . --fix

# コードフォーマット
uv run ruff format .

# 最終確認（エラーがないことを確認）
uv run ruff check .
uv run ruff format --check .
```

## uv の基本的な使い方

```bash
# パッケージの追加
uv add <package-name>

# 開発用パッケージの追加
uv add --dev <package-name>

# コマンドの実行
uv run <command>
```
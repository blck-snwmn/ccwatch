- YOU MUST: タスク完了またはユーザーにメッセージを返すときに最後に一度だけ `afplay /System/Library/Sounds/Funk.aiff -v 30 ` コマンドを実行して通知する

# ClaudeCode ログ可視化ツール「ccwatch」

## プロジェクト概要
ClaudeCodeのログファイル（~/.claude/projects/**/*.jsonl）を監視・可視化するWebアプリケーション。
DuckDBのJSONL直接読み込み機能を活用し、ゼロコピーで高速なクエリ処理を実現。

## 開発サイクルと動作確認フロー

### 1. エラー検知システム
アプリケーションには自律的なエラー検知システムが組み込まれています：
- エラーは `error.log` ファイルに自動的に記録される
- エラーログには、タイムスタンプ、エラータイプ、詳細なメッセージ、処理中のファイル情報が含まれる
- 定期的に `cat error.log` でエラーを確認する

### 2. Streamlitアプリの起動と動作確認
```bash
# 開発時の起動（ターミナルをブロックしない）
uv run streamlit run src/app.py &

# ヘッドレスモードでの起動（ログをファイルに保存）
uv run streamlit run src/app.py --server.headless true --server.port 8501 > streamlit.log 2>&1 &

# プロセスの確認
ps aux | grep streamlit

# プロセスの停止
pkill -f "streamlit run src/app.py"
```

### 3. DuckDBを使用したデータ構造の確認
JSONLファイルの構造を確認する際は、DuckDBのCLIを直接使用する：

```bash
# ファイルのスキーマを確認
duckdb -c "DESCRIBE SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 1;"

# 実際のデータを確認
duckdb -c "SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 5;"

# 特定のカラムの存在を確認
duckdb -c "SELECT column_name FROM (DESCRIBE SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited'));"
```

### 4. エラー解消の基本的なアプローチ
1. **エラーログの確認**: `tail -f error.log` でリアルタイムにエラーを監視
2. **DuckDBでスキーマ確認**: エラーの原因となっているカラムやデータ型を特定
3. **段階的な修正**: 
   - まず最小限の動作するクエリを作成
   - 徐々に機能を追加していく
   - 各段階でエラーログを確認

### 5. 効果的な開発サイクルの実践方法
- **段階的アプローチ**: 最小限の機能から始めて、動作確認後に機能を追加
- **自律的エラー検知**: error.logを活用し、ユーザー報告に依存しない問題発見
- **データ構造の事前確認**: DuckDB CLIで実際のスキーマを確認してからクエリを作成
- **継続的な動作確認**: 変更のたびにアプリを起動してエラーログを確認

### 6. タスク完了前の必須チェック項目
ユーザーへの完了報告を行う前に、必ず以下の手順を実行する：

1. **動作確認とエラー修正**
   ```bash
   # アプリケーションをヘッドレスモードで実行
   uv run streamlit run src/app.py --server.headless true --server.port 8501 > streamlit.log 2>&1 &
   
   # 起動を待つ
   sleep 3
   
   # エラーログを確認
   cat error.log
   
   # エラーがある場合は修正し、再度実行して確認
   ```

2. **コード品質チェック（Ruff）**
   ```bash
   # Lintチェックと自動修正
   uv run ruff check . --fix
   
   # コードフォーマット
   uv run ruff format .
   
   # 最終確認（エラーがないことを確認）
   uv run ruff check .
   uv run ruff format --check .
   ```

3. **プロセスのクリーンアップ**
   ```bash
   # 実行中のプロセスを停止
   pkill -f "streamlit run src/app.py"
   ```

これらすべてのチェックが完了し、エラーがないことを確認してから、ユーザーへの完了報告を行う。

## duckdb
実際に duckdb へクエリを実行し、どのような構造なのかなどの確認を行う必要があります。

## uv の使い方

### 基本コマンド
```bash
# 依存関係のインストール
uv sync

# パッケージの追加
uv add <package-name>

# 開発用パッケージの追加
uv add --dev <package-name>

# パッケージの削除
uv remove <package-name>

# コマンドの実行
uv run <command>

# Pythonスクリプトの実行
uv run python <script.py>

# pytestの実行
uv run pytest
```

### このプロジェクトでの使用例
```bash
# アプリケーションの起動
uv run streamlit run src/app.py

# テストの実行
uv run pytest

# 新しいパッケージの追加（例：新しい分析ライブラリ）
uv add plotly

# 開発ツールの追加
uv add --dev mypy ruff
```

## Ruff の使い方

### 基本的な実行方法
```bash
# コードのリント（問題の検出）
uv run ruff check .

# 特定のファイルをチェック
uv run ruff check src/app.py

# 自動修正可能な問題を修正
uv run ruff check --fix .

# より詳細な出力
uv run ruff check --verbose .

# フォーマット（コード整形）
uv run ruff format .

# フォーマットのチェックのみ（変更しない）
uv run ruff format --check .
```

### 推奨される使用フロー
1. **開発中の定期的なチェック**
   ```bash
   uv run ruff check . --fix
   uv run ruff format .
   ```

2. **コミット前の確認**
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```

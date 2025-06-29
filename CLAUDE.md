# ClaudeCode Log Visualization Tool "ccwatch"

## Project Overview
A web application that monitors and visualizes ClaudeCode log files (~/.claude/projects/**/*.jsonl).
Utilizes DuckDB's direct JSONL reading capability for zero-copy, high-performance query processing.

## Development Cycle

### Starting the Streamlit App
```bash
# Install dependencies
uv sync

# Start the application
uv run streamlit run src/app.py

# Start in background
uv run streamlit run src/app.py > streamlit.log 2>&1 &

# Stop process
pkill -f "streamlit run src/app.py"
```

### Checking Data Structure with DuckDB
```bash
# Check file schema
duckdb -c "DESCRIBE SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 1;"

# Check actual data
duckdb -c "SELECT * FROM read_json_auto('/path/to/file.jsonl', format='newline_delimited') LIMIT 5;"
```

### Error Checking
- Errors are automatically logged to `error.log`
- Check errors with `cat error.log`

## Code Quality Management (Ruff)

### Required Commands
```bash
# Lint check and auto-fix
uv run ruff check . --fix

# Code formatting
uv run ruff format .

# Final check (ensure no errors)
uv run ruff check .
uv run ruff format --check .
```

## Basic uv Usage

```bash
# Add package
uv add <package-name>

# Add development package
uv add --dev <package-name>

# Run command
uv run <command>
```
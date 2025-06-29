# ccwatch - ClaudeCode Log Visualization Tool

## Overview

ccwatch is a web application that loads and visualizes ClaudeCode usage logs (`~/.claude/projects/**/*.jsonl`) from various perspectives. It leverages DuckDB's high-performance JSONL reading capabilities to efficiently process and analyze large volumes of log data.

## Key Features

### 📊 Statistics Visualization
- **Hourly Message Count Trends**: Track when you use ClaudeCode throughout the day
- **Model Usage Distribution**: View AI model usage as pie charts
- **Annual Activity Heatmap**: GitHub-style heatmap showing yearly activity at a glance
- **Session Analysis**: Analyze session duration and message count distributions

### 🔍 Project Analysis
- Project-wise message count rankings
- Recently used projects list
- Filter by project selection in the sidebar

### 🔄 Auto Refresh
- Automatic refresh every 5 minutes
- Manual refresh button for instant data updates

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- ClaudeCode installed with existing log files
- uv package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/blck-snwmn/ccwatch.git
cd ccwatch

# Install dependencies
uv sync

# Start the application
uv run streamlit run src/app.py
```

The application will be accessible at http://localhost:8501 by default.

## Usage

1. Launch the application to automatically load log files from `~/.claude/projects/`
2. All information is displayed on a single page:
   - Overall statistics and graphs
   - Session analysis
   - Model usage
   - Annual activity heatmap
   - Project insights
   - Recent log entries

## Technology Stack

- **Frontend**: Streamlit (Python-based interactive web app framework)
- **Data Processing**: DuckDB (High-performance in-memory analytical database)
- **Visualization**: Plotly (Interactive graphing)
- **File Monitoring**: Watchdog
- **Development Tools**: uv, Ruff
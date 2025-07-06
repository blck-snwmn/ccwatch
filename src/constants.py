"""Constants used throughout the ccwatch application."""

# Session analysis constants
DEFAULT_IDLE_THRESHOLD_MINUTES = 30
SESSION_ACTIVITY_THRESHOLDS = {
    "quick_query": {"responses": 5, "minutes": None},
    "focused_task": {"responses": 20, "minutes": 15},
    "standard_session": {"responses": 50, "minutes": 30},
    "extended_session": {"responses": 100, "minutes": 60},
    "intensive_work": {"responses": 200, "minutes": 120},
    "high_intensity": {"responses": 500, "minutes": None},
    "marathon_session": {"responses": float("inf"), "minutes": 240},
}

# Display constants
MESSAGE_PREVIEW_LENGTH = 200
MAX_PROJECTS_TO_SHOW = 10
HEATMAP_DAYS = 364

# Chart constants
HOURLY_ACTIVITY_INTERVAL = "1h"
MINUTE_ACTIVITY_INTERVAL = "5min"
DAILY_ACTIVITY_INTERVAL = "D"

# Token calculation constants
CACHE_READ_DISCOUNT = 0.1  # Cache read tokens cost 10% of regular tokens
TOKENS_PER_MILLION = 1_000_000

# UI refresh constants
AUTOREFRESH_KEY = "autorefresh"
UPDATE_COUNT_KEY = "update_count"

# File operation constants
JSONL_FORMAT = "newline_delimited"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

# Logging constants
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Error message constants
ERROR_NO_DATA = "Failed to load data"
ERROR_NO_FILES = "No ClaudeCode log files found"
ERROR_CLAUDE_PATH_NOT_FOUND = "ClaudeCode projects directory not found: {}"

# Chart height constants
CHART_HEIGHT_DEFAULT = 400
CHART_HEIGHT_SMALL = 300
CHART_HEIGHT_LARGE = 500

# Session status constants
SESSION_STATUS_PENDING = "pending"
SESSION_STATUS_IN_PROGRESS = "in_progress"
SESSION_STATUS_COMPLETED = "completed"

"""Tests for logging functionality"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_config import JsonFormatter, get_logger, log_with_context, setup_logger


class TestJsonFormatter:
    """Test JSON formatter functionality"""

    def test_formats_log_as_json(self):
        """Test basic JSON formatting"""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO", f"Expected log level 'INFO' but got '{data['level']}'"
        assert data["message"] == "Test message", f"Expected message 'Test message' but got '{data['message']}'"
        assert data["module"] == "test", f"Expected module 'test' but got '{data['module']}'"
        # Function name might be None when creating LogRecord manually
        assert data["function"] in ["test_json_formatter_basic", "<module>", None]
        assert data["line"] == 10, f"Expected line number 10 but got {data['line']}"
        assert "timestamp" in data, "Timestamp field is missing from log data"
        assert "exception" not in data, "Exception field should not be present in normal log"

    def test_includes_extra_fields_in_json(self):
        """Test JSON formatting with extra data"""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=20,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"user_id": 123, "action": "test_action"}

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["user_id"] == 123, f"Expected user_id 123 but got {data['user_id']}"
        assert data["action"] == "test_action", f"Expected action 'test_action' but got '{data['action']}'"
        assert data["message"] == "Debug message", f"Expected message 'Debug message' but got '{data['message']}'"

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info"""
        formatter = JsonFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=30,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "ERROR", f"Expected log level 'ERROR' but got '{data['level']}'"
        assert data["message"] == "Error occurred", f"Expected message 'Error occurred' but got '{data['message']}'"
        assert "exception" in data, "Exception field is missing from error log"
        assert "ValueError: Test exception" in data["exception"], (
            "Exception traceback should contain ValueError message"
        )


class TestSetupLogger:
    """Test logger setup functionality"""

    def test_creates_file_logger_with_defaults(self, tmp_path, monkeypatch):
        """Test logger setup with default configuration"""
        # Change to temp directory to avoid creating logs in project
        monkeypatch.chdir(tmp_path)

        logger = setup_logger("test_logger")

        assert logger.name == "test_logger", f"Expected logger name 'test_logger' but got '{logger.name}'"
        assert logger.level == logging.INFO, f"Expected log level INFO ({logging.INFO}) but got {logger.level}"
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.handlers.RotatingFileHandler)

        # Check log file was created
        log_file = tmp_path / "logs" / "ccwatch.log"
        assert log_file.exists()

    def test_setup_logger_with_env_vars(self, tmp_path, monkeypatch):
        """Test logger setup with environment variables"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CCWATCH_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("CCWATCH_DEBUG", "true")

        logger = setup_logger("test_logger_env")

        assert logger.level == logging.DEBUG, f"Expected log level DEBUG ({logging.DEBUG}) but got {logger.level}"
        assert len(logger.handlers) == 2, f"Expected 2 handlers (file and console) but got {len(logger.handlers)}"

        # Find handler types
        handler_types = {type(h).__name__ for h in logger.handlers}
        assert "RotatingFileHandler" in handler_types
        assert "StreamHandler" in handler_types

    def test_no_duplicate_handlers_on_repeated_calls(self, tmp_path, monkeypatch):
        """Test that setup_logger doesn't add duplicate handlers"""
        monkeypatch.chdir(tmp_path)

        logger1 = setup_logger("test_idempotent")
        handler_count1 = len(logger1.handlers)

        logger2 = setup_logger("test_idempotent")
        handler_count2 = len(logger2.handlers)

        assert logger1 is logger2
        assert handler_count1 == handler_count2

    def test_setup_logger_creates_log_directory(self, tmp_path, monkeypatch):
        """Test that setup_logger creates log directory if it doesn't exist"""
        monkeypatch.chdir(tmp_path)

        # Ensure logs directory doesn't exist
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()

        setup_logger("test_dir_creation")

        # Check directory was created
        assert log_dir.exists()
        assert log_dir.is_dir()


class TestLogWithContext:
    """Test contextual logging functionality"""

    def test_log_with_context_basic(self, tmp_path, monkeypatch, caplog):
        """Test basic contextual logging"""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger("test_context")
        log_with_context(logger, "INFO", "Test message", user_id=123, action="test")

        # Read the log file
        log_file = tmp_path / "logs" / "ccwatch.log"
        with open(log_file) as f:
            log_content = f.read()

        log_data = json.loads(log_content.strip())
        assert log_data["message"] == "Test message", f"Expected message 'Test message' but got '{log_data['message']}'"
        assert log_data["user_id"] == 123, f"Expected user_id 123 but got {log_data['user_id']}"
        assert log_data["action"] == "test", f"Expected action 'test' but got '{log_data['action']}'"
        assert log_data["level"] == "INFO", f"Expected log level 'INFO' but got '{log_data['level']}'"

    def test_log_with_context_different_levels(self, tmp_path, monkeypatch):
        """Test contextual logging with different log levels"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CCWATCH_LOG_LEVEL", "DEBUG")

        logger = setup_logger("test_levels")

        # Log messages at different levels
        log_with_context(logger, "DEBUG", "Debug message", debug_info="test")
        log_with_context(logger, "INFO", "Info message", info_data="test")
        log_with_context(logger, "WARNING", "Warning message", warning_type="test")
        log_with_context(logger, "ERROR", "Error message", error_code=500)

        # Read all log entries
        log_file = tmp_path / "logs" / "ccwatch.log"
        with open(log_file) as f:
            log_lines = f.readlines()

        assert len(log_lines) == 4, f"Expected 4 log lines but got {len(log_lines)}"

        # Verify each log entry
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for _i, (line, expected_level) in enumerate(zip(log_lines, levels)):
            data = json.loads(line.strip())
            assert data["level"] == expected_level, f"Expected log level '{expected_level}' but got '{data['level']}'"

    def test_context_logging_without_extra_fields(self, tmp_path, monkeypatch):
        """Test contextual logging without extra data"""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger("test_no_extra")
        log_with_context(logger, "INFO", "Simple message")

        log_file = tmp_path / "logs" / "ccwatch.log"
        with open(log_file) as f:
            log_content = f.read()

        log_data = json.loads(log_content.strip())
        assert log_data["message"] == "Simple message"
        assert log_data["level"] == "INFO"
        # Should not have extra fields beyond standard ones
        standard_fields = {"timestamp", "level", "module", "function", "line", "message"}
        assert set(log_data.keys()) == standard_fields


class TestGetLogger:
    """Test get_logger helper function"""

    def test_get_logger_default(self, tmp_path, monkeypatch):
        """Test get_logger with default name"""
        monkeypatch.chdir(tmp_path)

        logger = get_logger()
        assert logger.name == "ccwatch"
        assert len(logger.handlers) > 0

    def test_get_logger_custom_name(self, tmp_path, monkeypatch):
        """Test get_logger with custom name"""
        monkeypatch.chdir(tmp_path)

        logger = get_logger("custom_logger")
        assert logger.name == "custom_logger"
        assert len(logger.handlers) > 0

    def test_get_logger_returns_same_instance(self, tmp_path, monkeypatch):
        """Test that get_logger returns the same logger instance"""
        monkeypatch.chdir(tmp_path)

        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")

        assert logger1 is logger2


class TestRotatingFileHandler:
    """Test log rotation functionality"""

    def test_log_rotation_settings(self, tmp_path, monkeypatch):
        """Test that RotatingFileHandler is configured correctly"""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger("test_rotation")
        handler = logger.handlers[0]

        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert handler.backupCount == 5

    def test_log_file_encoding(self, tmp_path, monkeypatch):
        """Test that log files use UTF-8 encoding"""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger("test_encoding")
        log_with_context(logger, "INFO", "Test with unicode: こんにちは", emoji="👍")

        log_file = tmp_path / "logs" / "ccwatch.log"
        with open(log_file, encoding="utf-8") as f:
            content = f.read()

        data = json.loads(content.strip())
        assert "こんにちは" in data["message"]
        assert data["emoji"] == "👍"

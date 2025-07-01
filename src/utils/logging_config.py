import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """JSON形式でログを出力するカスタムフォーマッター"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # extra属性がある場合は追加
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # エラーの場合は例外情報を追加
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name: str = "ccwatch") -> logging.Logger:
    """アプリケーション用のロガーをセットアップ"""
    logger = logging.getLogger(name)

    # 既にハンドラーが設定されている場合はスキップ
    if logger.handlers:
        return logger

    log_level = os.getenv("CCWATCH_LOG_LEVEL", "INFO")  # Default to INFO
    logger.setLevel(getattr(logging, log_level.upper()))

    # ログディレクトリの作成
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ファイルハンドラーの設定
    file_handler = RotatingFileHandler(
        log_dir / "ccwatch.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # デバッグモードの場合はコンソール出力も追加
    if os.getenv("CCWATCH_DEBUG", "false").lower() == "true":
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs: Any) -> None:
    """コンテキスト情報付きでログを記録"""
    # ログレベルの取得
    log_level = getattr(logging, level.upper())

    # LogRecordにextra_dataを追加
    extra = {"extra_data": kwargs} if kwargs else {}

    logger.log(log_level, message, extra=extra)


# ヘルパー関数
def get_logger(name: str = "ccwatch") -> logging.Logger:
    """既存のロガーを取得または新規作成"""
    return setup_logger(name)

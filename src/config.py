"""Application configuration management"""

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration settings"""

    # Path settings
    claude_projects_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "projects")
    jsonl_pattern: str = "**/*.jsonl"

    # Display settings
    max_projects_to_show: int = 10
    check_interval: int = 300  # seconds
    message_preview_length: int = 100

    # Cache settings
    cache_ttl: int = 3600  # seconds

    # Model pricing information (per 1M tokens)
    # Source: https://docs.anthropic.com/en/docs/about-claude/models (2025-01)
    model_pricing: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            # Claude Sonnet 3.5 / 3.7
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
            "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
            # Claude Opus 3
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
            # Claude Sonnet 3
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
            # Claude Haiku 3.5
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "cache_read": 0.08},
            # Claude Haiku 3
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cache_read": 0.03},
            # Claude Opus 4
            "claude-opus-4-20250514": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
            # Claude Sonnet 4 (if model ID exists)
            "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
            # Default pricing for unknown models
            "default": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
        }
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        config = cls()

        # Override with environment variables if present
        if claude_path := os.getenv("CLAUDE_PROJECTS_PATH"):
            config.claude_projects_path = Path(claude_path)

        if jsonl_pattern := os.getenv("JSONL_PATTERN"):
            config.jsonl_pattern = jsonl_pattern

        if max_projects := os.getenv("MAX_PROJECTS_TO_SHOW"):
            with contextlib.suppress(ValueError):
                config.max_projects_to_show = int(max_projects)

        if check_interval := os.getenv("CHECK_INTERVAL"):
            with contextlib.suppress(ValueError):
                config.check_interval = int(check_interval)

        if message_preview := os.getenv("MESSAGE_PREVIEW_LENGTH"):
            with contextlib.suppress(ValueError):
                config.message_preview_length = int(message_preview)

        if cache_ttl := os.getenv("CACHE_TTL"):
            with contextlib.suppress(ValueError):
                config.cache_ttl = int(cache_ttl)

        return config

    def get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a specific model, with fallback to default"""
        return self.model_pricing.get(model, self.model_pricing["default"])

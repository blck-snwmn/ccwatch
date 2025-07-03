"""Tests for AppConfig"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AppConfig


class TestAppConfig:
    """Test AppConfig functionality"""

    def test_default_config(self):
        """Test default configuration values"""
        config = AppConfig()

        # Path settings
        assert config.claude_projects_path == Path.home() / ".claude" / "projects"
        assert config.jsonl_pattern == "**/*.jsonl"

        # Display settings
        assert config.max_projects_to_show == 10
        assert config.check_interval == 300
        assert config.message_preview_length == 100

        # Cache settings
        assert config.cache_ttl == 3600

        # Model pricing
        assert isinstance(config.model_pricing, dict)
        assert "claude-3-5-sonnet-20241022" in config.model_pricing
        assert "default" in config.model_pricing

    def test_from_env_with_valid_values(self, mock_env_vars):
        """Test configuration from environment variables"""
        mock_env_vars(
            CLAUDE_PROJECTS_PATH="/custom/path",
            JSONL_PATTERN="*.jsonl",
            MAX_PROJECTS_TO_SHOW="20",
            CHECK_INTERVAL="600",
            MESSAGE_PREVIEW_LENGTH="200",
            CACHE_TTL="7200",
        )

        config = AppConfig.from_env()

        assert config.claude_projects_path == Path("/custom/path")
        assert config.jsonl_pattern == "*.jsonl"
        assert config.max_projects_to_show == 20
        assert config.check_interval == 600
        assert config.message_preview_length == 200
        assert config.cache_ttl == 7200

    def test_from_env_with_invalid_values(self, mock_env_vars):
        """Test configuration with invalid environment variable values"""
        mock_env_vars(
            MAX_PROJECTS_TO_SHOW="invalid",
            CHECK_INTERVAL="not_a_number",
            MESSAGE_PREVIEW_LENGTH="",
            CACHE_TTL="abc",
        )

        config = AppConfig.from_env()

        # Invalid values should fall back to defaults
        assert config.max_projects_to_show == 10
        assert config.check_interval == 300
        assert config.message_preview_length == 100
        assert config.cache_ttl == 3600

    def test_from_env_partial_override(self, mock_env_vars):
        """Test partial environment variable override"""
        mock_env_vars(
            CLAUDE_PROJECTS_PATH="/partial/path",
            MAX_PROJECTS_TO_SHOW="15",
        )

        config = AppConfig.from_env()

        # Overridden values
        assert config.claude_projects_path == Path("/partial/path")
        assert config.max_projects_to_show == 15

        # Default values
        assert config.jsonl_pattern == "**/*.jsonl"
        assert config.check_interval == 300
        assert config.message_preview_length == 100
        assert config.cache_ttl == 3600

    def test_get_model_pricing_known_models(self):
        """Test getting pricing for known models"""
        config = AppConfig()

        # Test various known models
        sonnet_pricing = config.get_model_pricing("claude-3-5-sonnet-20241022")
        assert sonnet_pricing["input"] == 3.00
        assert sonnet_pricing["output"] == 15.00
        assert sonnet_pricing["cache_read"] == 0.30

        opus_pricing = config.get_model_pricing("claude-3-opus-20240229")
        assert opus_pricing["input"] == 15.00
        assert opus_pricing["output"] == 75.00
        assert opus_pricing["cache_read"] == 1.50

        haiku_pricing = config.get_model_pricing("claude-3-haiku-20240307")
        assert haiku_pricing["input"] == 0.25
        assert haiku_pricing["output"] == 1.25
        assert haiku_pricing["cache_read"] == 0.03

    def test_get_model_pricing_unknown_model(self):
        """Test getting pricing for unknown model falls back to default"""
        config = AppConfig()

        unknown_pricing = config.get_model_pricing("claude-unknown-model-xyz")
        assert unknown_pricing == config.model_pricing["default"]
        assert unknown_pricing["input"] == 3.00
        assert unknown_pricing["output"] == 15.00
        assert unknown_pricing["cache_read"] == 0.30

    def test_model_pricing_completeness(self):
        """Test that all model pricing entries have required fields"""
        config = AppConfig()

        required_fields = {"input", "output", "cache_read"}

        for model_name, pricing in config.model_pricing.items():
            assert isinstance(pricing, dict), f"Pricing for {model_name} should be a dict"
            assert set(pricing.keys()) == required_fields, f"Pricing for {model_name} missing required fields"

            # All values should be positive floats
            for field, value in pricing.items():
                assert isinstance(value, (int, float)), f"{field} for {model_name} should be numeric"
                assert value > 0, f"{field} for {model_name} should be positive"

    @pytest.mark.parametrize(
        "model,expected_input,expected_output,expected_cache",
        [
            ("claude-3-5-sonnet-20241022", 3.00, 15.00, 0.30),
            ("claude-3-5-sonnet-20240620", 3.00, 15.00, 0.30),
            ("claude-3-opus-20240229", 15.00, 75.00, 1.50),
            ("claude-3-sonnet-20240229", 3.00, 15.00, 0.30),
            ("claude-3-5-haiku-20241022", 0.80, 4.00, 0.08),
            ("claude-3-haiku-20240307", 0.25, 1.25, 0.03),
            ("claude-opus-4-20250514", 15.00, 75.00, 1.50),
            ("claude-sonnet-4", 3.00, 15.00, 0.30),
            ("unknown-model", 3.00, 15.00, 0.30),  # Should use default
        ],
    )
    def test_model_pricing_values(self, model, expected_input, expected_output, expected_cache):
        """Test specific model pricing values"""
        config = AppConfig()
        pricing = config.get_model_pricing(model)

        assert pricing["input"] == expected_input
        assert pricing["output"] == expected_output
        assert pricing["cache_read"] == expected_cache

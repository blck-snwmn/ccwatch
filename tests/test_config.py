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
        assert config.claude_projects_path == Path.home() / ".claude" / "projects", (
            f"Expected default Claude projects path but got {config.claude_projects_path}"
        )
        assert config.jsonl_pattern == "**/*.jsonl", (
            f"Expected default JSONL pattern '**/*.jsonl' but got '{config.jsonl_pattern}'"
        )

        # Display settings
        assert config.max_projects_to_show == 10, (
            f"Expected max_projects_to_show=10 but got {config.max_projects_to_show}"
        )
        assert config.check_interval == 300, f"Expected check_interval=300 but got {config.check_interval}"
        assert config.message_preview_length == 100, (
            f"Expected message_preview_length=100 but got {config.message_preview_length}"
        )

        # Cache settings
        assert config.cache_ttl == 3600, f"Expected cache_ttl=3600 but got {config.cache_ttl}"

        # Model pricing
        assert isinstance(config.model_pricing, dict), (
            f"Expected model_pricing to be dict but got {type(config.model_pricing)}"
        )
        assert "claude-3-5-sonnet-20241022" in config.model_pricing, "Sonnet model pricing not found in config"
        assert "default" in config.model_pricing, "Default model pricing not found in config"

    def test_load_config_from_environment_variables(self, mock_env_vars):
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

        assert config.claude_projects_path == Path("/custom/path"), (
            f"Expected custom path but got {config.claude_projects_path}"
        )
        assert config.jsonl_pattern == "*.jsonl", f"Expected custom pattern '*.jsonl' but got '{config.jsonl_pattern}'"
        assert config.max_projects_to_show == 20, (
            f"Expected max_projects_to_show=20 but got {config.max_projects_to_show}"
        )
        assert config.check_interval == 600, f"Expected check_interval=600 but got {config.check_interval}"
        assert config.message_preview_length == 200, (
            f"Expected message_preview_length=200 but got {config.message_preview_length}"
        )
        assert config.cache_ttl == 7200, f"Expected cache_ttl=7200 but got {config.cache_ttl}"

    def test_invalid_env_vars_fallback_to_defaults(self, mock_env_vars):
        """Test configuration with invalid environment variable values"""
        mock_env_vars(
            MAX_PROJECTS_TO_SHOW="invalid",
            CHECK_INTERVAL="not_a_number",
            MESSAGE_PREVIEW_LENGTH="",
            CACHE_TTL="abc",
        )

        config = AppConfig.from_env()

        # Invalid values should fall back to defaults
        assert config.max_projects_to_show == 10, "Invalid max_projects_to_show should fall back to default 10"
        assert config.check_interval == 300, "Invalid check_interval should fall back to default 300"
        assert config.message_preview_length == 100, "Invalid message_preview_length should fall back to default 100"
        assert config.cache_ttl == 3600, "Invalid cache_ttl should fall back to default 3600"

    def test_partial_env_override_keeps_other_defaults(self, mock_env_vars):
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

    def test_pricing_for_known_models(self):
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

    def test_unknown_model_uses_default_pricing(self):
        """Test getting pricing for unknown model falls back to default"""
        config = AppConfig()

        unknown_pricing = config.get_model_pricing("claude-unknown-model-xyz")
        assert unknown_pricing == config.model_pricing["default"], "Unknown model should use default pricing"
        assert unknown_pricing["input"] == 3.00, (
            f"Expected default input price $3.00 but got ${unknown_pricing['input']}"
        )
        assert unknown_pricing["output"] == 15.00, (
            f"Expected default output price $15.00 but got ${unknown_pricing['output']}"
        )
        assert unknown_pricing["cache_read"] == 0.30, (
            f"Expected default cache read price $0.30 but got ${unknown_pricing['cache_read']}"
        )

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
        ("model", "expected_input", "expected_output", "expected_cache"),
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

        assert pricing["input"] == expected_input, (
            f"Model {model}: expected input price ${expected_input} but got ${pricing['input']}"
        )
        assert pricing["output"] == expected_output, (
            f"Model {model}: expected output price ${expected_output} but got ${pricing['output']}"
        )
        assert pricing["cache_read"] == expected_cache, (
            f"Model {model}: expected cache read price ${expected_cache} but got ${pricing['cache_read']}"
        )

    def test_config_with_extreme_values(self, mock_env_vars):
        """Test configuration with extreme boundary values"""
        mock_env_vars(
            MAX_PROJECTS_TO_SHOW="999999999",
            CHECK_INTERVAL="0",  # Zero interval
            MESSAGE_PREVIEW_LENGTH="10000000",  # Very large preview
            CACHE_TTL="-1",  # Negative TTL
        )

        config = AppConfig.from_env()

        # Large values should be accepted
        assert config.max_projects_to_show == 999999999, (
            f"Expected max_projects_to_show=999999999 but got {config.max_projects_to_show}"
        )
        assert config.message_preview_length == 10000000, (
            f"Expected message_preview_length=10000000 but got {config.message_preview_length}"
        )

        # Numeric values are accepted even if they're extreme (current behavior)
        assert config.check_interval == 0, "Zero check_interval is accepted (current behavior)"
        assert config.cache_ttl == -1, "Negative cache_ttl is accepted (current behavior)"

    def test_empty_model_name_uses_default_pricing(self):
        """Test that empty model name falls back to default pricing"""
        config = AppConfig()

        pricing = config.get_model_pricing("")
        assert pricing == config.model_pricing["default"], "Empty model name should use default pricing"
        assert pricing["input"] == 3.00, f"Expected default input price $3.00 but got ${pricing['input']}"

    def test_very_long_model_name(self):
        """Test configuration with extremely long model name"""
        config = AppConfig()

        # Create a very long model name (1000+ characters)
        long_model_name = "claude-" + "x" * 1000 + "-model"
        pricing = config.get_model_pricing(long_model_name)

        assert pricing == config.model_pricing["default"], "Very long unknown model name should use default pricing"

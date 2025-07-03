"""Tests for calculation functions"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import calculate_cost
from config import AppConfig


class TestCalculateCost:
    """Test cost calculation functionality"""

    def test_cost_calculation_for_sonnet(self):
        """Test cost calculation for Sonnet model"""

        # Create test row with Sonnet model
        row = pd.Series(
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": 1000000,  # 1M tokens
                "cache_creation_input_tokens": 500000,  # 0.5M tokens
                "cache_read_input_tokens": 200000,  # 0.2M tokens
                "output_tokens": 300000,  # 0.3M tokens
            }
        )

        cost = calculate_cost(row)

        # Expected costs (per 1M tokens):
        # Input: (1M + 0.5M) * $3.00 = $4.50
        # Cache read: 0.2M * $0.30 = $0.06
        # Output: 0.3M * $15.00 = $4.50
        # Total: $4.50 + $0.06 + $4.50 = $9.06
        expected_cost = 4.50 + 0.06 + 4.50
        assert cost == pytest.approx(expected_cost, rel=1e-5), (
            f"Expected cost ${expected_cost:.2f} but got ${cost:.2f} for Sonnet model"
        )

    def test_cost_calculation_for_opus(self):
        """Test cost calculation for Opus model"""

        row = pd.Series(
            {
                "model": "claude-3-opus-20240229",
                "input_tokens": 100000,  # 0.1M tokens
                "cache_creation_input_tokens": 50000,  # 0.05M tokens
                "cache_read_input_tokens": 20000,  # 0.02M tokens
                "output_tokens": 30000,  # 0.03M tokens
            }
        )

        cost = calculate_cost(row)

        # Expected costs (per 1M tokens):
        # Input: (0.1M + 0.05M) * $15.00 = $2.25
        # Cache read: 0.02M * $1.50 = $0.03
        # Output: 0.03M * $75.00 = $2.25
        # Total: $2.25 + $0.03 + $2.25 = $4.53
        expected_cost = 2.25 + 0.03 + 2.25
        assert cost == pytest.approx(expected_cost, rel=1e-5), (
            f"Expected cost ${expected_cost:.2f} but got ${cost:.2f} for Opus model"
        )

    def test_cost_calculation_for_haiku(self):
        """Test cost calculation for Haiku model"""

        row = pd.Series(
            {
                "model": "claude-3-haiku-20240307",
                "input_tokens": 1000000,  # 1M tokens
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1000000,  # 1M tokens
                "output_tokens": 500000,  # 0.5M tokens
            }
        )

        cost = calculate_cost(row)

        # Expected costs (per 1M tokens):
        # Input: 1M * $0.25 = $0.25
        # Cache read: 1M * $0.03 = $0.03
        # Output: 0.5M * $1.25 = $0.625
        # Total: $0.25 + $0.03 + $0.625 = $0.905
        expected_cost = 0.25 + 0.03 + 0.625
        assert cost == pytest.approx(expected_cost, rel=1e-5), (
            f"Expected cost ${expected_cost:.2f} but got ${cost:.2f} for Haiku model"
        )

    def test_unknown_model_uses_default_pricing(self):
        """Test cost calculation for unknown model uses default pricing"""

        row = pd.Series(
            {
                "model": "claude-unknown-model",
                "input_tokens": 100000,  # 0.1M tokens
                "cache_creation_input_tokens": 50000,  # 0.05M tokens
                "cache_read_input_tokens": 20000,  # 0.02M tokens
                "output_tokens": 30000,  # 0.03M tokens
            }
        )

        cost = calculate_cost(row)

        # Should use default pricing (same as Sonnet)
        # Input: (0.1M + 0.05M) * $3.00 = $0.45
        # Cache read: 0.02M * $0.30 = $0.006
        # Output: 0.03M * $15.00 = $0.45
        # Total: $0.45 + $0.006 + $0.45 = $0.906
        expected_cost = 0.45 + 0.006 + 0.45
        assert cost == pytest.approx(expected_cost, rel=1e-5), (
            f"Expected cost ${expected_cost:.2f} but got ${cost:.2f} for unknown model (should use default pricing)"
        )

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens"""

        row = pd.Series(
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            }
        )

        cost = calculate_cost(row)
        assert cost == 0.0, f"Expected zero cost for zero tokens but got ${cost:.2f}"

    def test_cost_with_input_tokens_only(self):
        """Test cost calculation with only input tokens"""

        row = pd.Series(
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": 1000000,  # 1M tokens
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            }
        )

        cost = calculate_cost(row)

        # Only input cost: 1M * $3.00 = $3.00
        assert cost == pytest.approx(3.00, rel=1e-5), f"Expected cost $3.00 for input tokens only but got ${cost:.2f}"

    def test_cost_with_cache_tokens_only(self):
        """Test cost calculation with only cache tokens"""

        row = pd.Series(
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": 0,
                "cache_creation_input_tokens": 1000000,  # 1M tokens (charged as input)
                "cache_read_input_tokens": 1000000,  # 1M tokens (10% cost)
                "output_tokens": 0,
            }
        )

        cost = calculate_cost(row)

        # Cache creation: 1M * $3.00 = $3.00
        # Cache read: 1M * $0.30 = $0.30
        # Total: $3.00 + $0.30 = $3.30
        expected_cost = 3.00 + 0.30
        assert cost == pytest.approx(expected_cost, rel=1e-5), (
            f"Expected cost ${expected_cost:.2f} for cache tokens only but got ${cost:.2f}"
        )

    @pytest.mark.parametrize(
        ("tokens", "expected_cost"),
        [
            # Small token counts
            ({"input": 100, "cache_creation": 50, "cache_read": 20, "output": 30}, 0.000906),
            # Large token counts
            ({"input": 10000000, "cache_creation": 5000000, "cache_read": 2000000, "output": 3000000}, 90.6),
            # Only output tokens
            ({"input": 0, "cache_creation": 0, "cache_read": 0, "output": 100000}, 1.5),
        ],
    )
    def test_cost_with_various_token_amounts(self, tokens, expected_cost):
        """Test cost calculation with various token amounts"""

        row = pd.Series(
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": tokens["input"],
                "cache_creation_input_tokens": tokens["cache_creation"],
                "cache_read_input_tokens": tokens["cache_read"],
                "output_tokens": tokens["output"],
            }
        )

        cost = calculate_cost(row)
        assert cost == pytest.approx(expected_cost, rel=1e-3), (
            f"Expected cost ${expected_cost:.6f} but got ${cost:.6f} for token amounts {tokens}"
        )

    def test_cost_calculation_for_all_configured_models(self):
        """Test cost calculation works for all configured models"""
        config = AppConfig()

        # Test row with standard token counts
        base_row = {
            "input_tokens": 100000,
            "cache_creation_input_tokens": 50000,
            "cache_read_input_tokens": 20000,
            "output_tokens": 30000,
        }

        for model_name in config.model_pricing:
            if model_name == "default":
                continue

            row = pd.Series({**base_row, "model": model_name})
            cost = calculate_cost(row)

            # Cost should be positive for any model with tokens
            assert cost > 0, f"Cost for {model_name} should be positive but got ${cost:.2f}"

    def test_calculate_cost_precision(self):
        """Test cost calculation maintains appropriate precision"""

        # Test with very small token counts
        row = pd.Series(
            {
                "model": "claude-3-haiku-20240307",  # Cheapest model
                "input_tokens": 1,
                "cache_creation_input_tokens": 1,
                "cache_read_input_tokens": 1,
                "output_tokens": 1,
            }
        )

        cost = calculate_cost(row)

        # Expected cost for 4 tokens with Haiku pricing
        # Input: 2 * $0.25 / 1M = $0.0000005
        # Cache read: 1 * $0.03 / 1M = $0.00000003
        # Output: 1 * $1.25 / 1M = $0.00000125
        # Total: ~$0.00000178
        assert cost > 0, f"Expected positive cost for minimal tokens but got ${cost:.8f}"
        assert cost < 0.001, f"Expected cost < $0.001 for minimal tokens but got ${cost:.8f}"

    def test_batch_cost_calculation_on_dataframe(self):
        """Test cost calculation on entire DataFrame"""
        # Create test DataFrame
        data = [
            {
                "model": "claude-3-5-sonnet-20241022",
                "input_tokens": 1000,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
                "output_tokens": 500,
            },
            {
                "model": "claude-3-opus-20240229",
                "input_tokens": 2000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model": "claude-3-haiku-20240307",
                "input_tokens": 5000,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 0,
                "output_tokens": 2000,
            },
        ]
        df = pd.DataFrame(data)

        # Apply cost calculation
        df["cost"] = df.apply(calculate_cost, axis=1)

        # All costs should be positive
        assert all(df["cost"] > 0), f"All costs should be positive but found: {df[df['cost'] <= 0]['cost'].tolist()}"

        # Verify relative costs (Opus should be most expensive per token)
        opus_row = df[df["model"].str.contains("opus")].iloc[0]
        sonnet_row = df[df["model"].str.contains("sonnet")].iloc[0]
        haiku_row = df[df["model"].str.contains("haiku")].iloc[0]

        # Calculate cost per total token for comparison
        opus_cost_per_token = opus_row["cost"] / (
            opus_row["input_tokens"]
            + opus_row["cache_creation_input_tokens"]
            + opus_row["cache_read_input_tokens"]
            + opus_row["output_tokens"]
        )
        sonnet_cost_per_token = sonnet_row["cost"] / (
            sonnet_row["input_tokens"]
            + sonnet_row["cache_creation_input_tokens"]
            + sonnet_row["cache_read_input_tokens"]
            + sonnet_row["output_tokens"]
        )
        haiku_cost_per_token = haiku_row["cost"] / (
            haiku_row["input_tokens"]
            + haiku_row["cache_creation_input_tokens"]
            + haiku_row["cache_read_input_tokens"]
            + haiku_row["output_tokens"]
        )

        # Opus should be most expensive, Haiku cheapest
        assert opus_cost_per_token > sonnet_cost_per_token, (
            f"Opus (${opus_cost_per_token:.6f}/token) should be more expensive than Sonnet (${sonnet_cost_per_token:.6f}/token)"
        )
        assert sonnet_cost_per_token > haiku_cost_per_token, (
            f"Sonnet (${sonnet_cost_per_token:.6f}/token) should be more expensive than Haiku (${haiku_cost_per_token:.6f}/token)"
        )

    def test_cost_with_maximum_safe_integer_tokens(self):
        """Test cost calculation with very large token counts (MAX_SAFE_INTEGER boundary)"""
        # JavaScript's MAX_SAFE_INTEGER is 2^53 - 1
        max_safe_int = 9007199254740991

        row = pd.Series(
            {
                "model": "claude-3-haiku-20240307",  # Use cheapest model
                "input_tokens": max_safe_int,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            }
        )

        cost = calculate_cost(row)
        # max_safe_int / 1M * $0.25 = 2,251,799,813.685248
        expected_cost = max_safe_int / 1_000_000 * 0.25
        assert cost == pytest.approx(expected_cost, rel=1e-10), (
            f"Expected cost ${expected_cost:.2f} for MAX_SAFE_INTEGER tokens but got ${cost:.2f}"
        )

    def test_empty_dataframe_cost_calculation(self):
        """Test cost calculation on empty DataFrame (boundary case)"""
        df = pd.DataFrame(
            columns=[
                "model",
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            ]
        )

        # Should not raise error on empty DataFrame
        df["cost"] = df.apply(calculate_cost, axis=1)
        assert len(df) == 0, "Empty DataFrame should remain empty"
        assert "cost" in df.columns, "Cost column should be added even to empty DataFrame"

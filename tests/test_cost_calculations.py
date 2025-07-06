"""Additional tests for cost calculation functions"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculations.cost import (
    aggregate_cost_by_model,
    aggregate_cost_by_time,
    calculate_cache_savings,
    calculate_cost_metrics,
    calculate_daily_average_cost,
)


class TestCalculateCacheSavings:
    """Test cache savings calculation"""

    def test_cache_savings_no_cache_read(self):
        """Test when there are no cache reads"""
        result = calculate_cache_savings(cache_read=0, regular_input=1000, cache_creation=500)
        assert result == 0.0

    def test_cache_savings_with_cache_read(self):
        """Test cache savings with cache reads"""
        # cache_read=1000, total=2000, savings=(1000 * 0.9 / 2000) * 100 = 45%
        result = calculate_cache_savings(cache_read=1000, regular_input=500, cache_creation=500)
        assert result == 45.0

    def test_cache_savings_all_cache_read(self):
        """Test when all tokens are cache reads"""
        result = calculate_cache_savings(cache_read=1000, regular_input=0, cache_creation=0)
        assert result == 90.0  # 90% savings

    def test_cache_savings_zero_tokens(self):
        """Test with zero total tokens"""
        result = calculate_cache_savings(cache_read=0, regular_input=0, cache_creation=0)
        assert result == 0.0

    def test_cache_savings_mixed_tokens(self):
        """Test with mixed token types"""
        # cache_read=2000, total=5000, savings=(2000 * 0.9 / 5000) * 100 = 36%
        result = calculate_cache_savings(cache_read=2000, regular_input=2000, cache_creation=1000)
        assert result == 36.0


class TestCalculateDailyAverageCost:
    """Test daily average cost calculation"""

    def test_daily_average_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = calculate_daily_average_cost(df, 100.0)
        assert result == 0.0

    def test_daily_average_no_timestamp_column(self):
        """Test with DataFrame missing timestamp column"""
        df = pd.DataFrame({"cost": [10, 20, 30]})
        result = calculate_daily_average_cost(df, 60.0)
        assert result == 0.0

    def test_daily_average_single_day(self):
        """Test with data from single day"""
        df = pd.DataFrame({"timestamp": [datetime.now(timezone.utc)] * 5})
        result = calculate_daily_average_cost(df, 100.0)
        assert result == 100.0  # 100 / 1 day

    def test_daily_average_multiple_days(self):
        """Test with data spanning multiple days"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=2),
                    base_time + timedelta(days=4),
                ]
            }
        )
        result = calculate_daily_average_cost(df, 100.0)
        assert result == 25.0  # 100 / 4 days

    def test_daily_average_within_same_day(self):
        """Test with timestamps within the same day"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(hours=6),
                    base_time + timedelta(hours=12),
                ]
            }
        )
        result = calculate_daily_average_cost(df, 90.0)
        assert result == 90.0  # All within same day


class TestAggregateCostByModel:
    """Test model-based cost aggregation"""

    def test_aggregate_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = aggregate_cost_by_model(df)
        assert result.empty

    def test_aggregate_with_existing_cost_column(self):
        """Test aggregation when cost column already exists"""
        df = pd.DataFrame(
            {
                "model": ["claude-3-opus", "claude-3-opus", "claude-3-sonnet"],
                "cost": [1.5, 2.5, 0.5],
                "effective_input_tokens": [1000, 2000, 500],
                "output_tokens": [500, 1000, 250],
                "input_tokens": [800, 1600, 400],
                "cache_creation_input_tokens": [200, 400, 100],
                "cache_read_input_tokens": [0, 0, 0],
            }
        )

        result = aggregate_cost_by_model(df)

        assert len(result) == 2
        assert result.loc["claude-3-opus", "cost"] == 4.0
        assert result.loc["claude-3-sonnet", "cost"] == 0.5
        assert result.loc["claude-3-opus", "cost_percentage"] == 88.9  # 4.0/4.5*100
        assert result.loc["claude-3-sonnet", "cost_percentage"] == 11.1  # 0.5/4.5*100

    def test_aggregate_without_cost_column(self):
        """Test aggregation when cost column needs to be calculated"""
        df = pd.DataFrame(
            {
                "model": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"],
                "input_tokens": [1000, 2000],
                "cache_creation_input_tokens": [0, 0],
                "cache_read_input_tokens": [0, 0],
                "output_tokens": [500, 1000],
                "effective_input_tokens": [1000, 2000],
            }
        )

        result = aggregate_cost_by_model(df)

        assert len(result) == 1
        assert "cost" in result.columns
        assert "cost_percentage" in result.columns
        assert result["cost_percentage"].iloc[0] == 100.0

    def test_aggregate_with_zero_total_cost(self):
        """Test when total cost is zero"""
        df = pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "cost": [0.0, 0.0],
                "effective_input_tokens": [0, 0],
                "output_tokens": [0, 0],
                "input_tokens": [0, 0],
                "cache_creation_input_tokens": [0, 0],
                "cache_read_input_tokens": [0, 0],
            }
        )

        result = aggregate_cost_by_model(df)

        assert result["cost_percentage"].sum() == 0.0


class TestAggregateCostByTime:
    """Test time-based cost aggregation"""

    def test_aggregate_time_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = aggregate_cost_by_time(df)
        assert result.empty

    def test_aggregate_time_no_timestamp(self):
        """Test with DataFrame missing timestamp"""
        df = pd.DataFrame({"cost": [1, 2, 3]})
        result = aggregate_cost_by_time(df)
        assert result.empty

    def test_aggregate_time_daily(self):
        """Test daily aggregation"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(hours=12),
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=1, hours=6),
                ],
                "cost": [10.0, 20.0, 15.0, 25.0],
            }
        )

        result = aggregate_cost_by_time(df, freq="D")

        assert len(result) == 2
        assert result["cost"].tolist() == [30.0, 40.0]  # Day 1: 10+20, Day 2: 15+25

    def test_aggregate_time_weekly(self):
        """Test weekly aggregation"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame({"timestamp": pd.date_range(base_time, periods=14, freq="D"), "cost": [5.0] * 14})

        result = aggregate_cost_by_time(df, freq="W")

        # Depending on start date, could be 2 or 3 weeks
        assert len(result) >= 2
        assert all(cost > 0 for cost in result["cost"])
        assert result["cost"].sum() == 70.0  # 14 * 5.0

    def test_aggregate_time_without_cost_column(self):
        """Test when cost needs to be calculated"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [base_time, base_time + timedelta(days=1)],
                "model": ["claude-3-5-sonnet-20241022"] * 2,
                "input_tokens": [1000, 2000],
                "cache_creation_input_tokens": [0, 0],
                "cache_read_input_tokens": [0, 0],
                "output_tokens": [500, 1000],
            }
        )

        result = aggregate_cost_by_time(df, freq="D")

        assert len(result) == 2
        assert "cost" in result.columns


class TestCalculateCostMetrics:
    """Test cost metrics calculation"""

    def test_metrics_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = calculate_cost_metrics(df)

        assert result["total_cost"] == 0.0
        assert result["daily_avg_cost"] == 0.0
        assert result["cache_hit_rate"] == 0.0
        assert result["avg_cost_per_response"] == 0.0

    def test_metrics_with_cost_column(self):
        """Test metrics when cost column exists"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [base_time, base_time + timedelta(days=1)],
                "cost": [50.0, 75.0],
                "effective_input_tokens": [1000, 2000],
                "cache_read_input_tokens": [100, 200],
            }
        )

        result = calculate_cost_metrics(df)

        assert result["total_cost"] == 125.0
        assert result["daily_avg_cost"] == 125.0  # 125/1 day
        assert result["cache_hit_rate"] == 10.0  # 300/3000*100
        assert result["avg_cost_per_response"] == 62.5  # 125/2

    def test_metrics_without_cost_column(self):
        """Test metrics when cost needs to be calculated"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [base_time] * 3,
                "model": ["claude-3-5-sonnet-20241022"] * 3,
                "input_tokens": [1000, 2000, 3000],
                "cache_creation_input_tokens": [0, 0, 0],
                "cache_read_input_tokens": [0, 0, 0],
                "output_tokens": [500, 1000, 1500],
                "effective_input_tokens": [1000, 2000, 3000],
            }
        )

        result = calculate_cost_metrics(df)

        assert result["total_cost"] > 0
        assert result["daily_avg_cost"] == result["total_cost"]  # Single day
        assert result["cache_hit_rate"] == 0.0
        assert result["avg_cost_per_response"] == result["total_cost"] / 3

    def test_metrics_without_effective_input_tokens(self):
        """Test cache hit rate when effective_input_tokens is missing"""
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(timezone.utc)],
                "cost": [100.0],
                "cache_read_input_tokens": [500],
            }
        )

        result = calculate_cost_metrics(df)

        assert result["cache_hit_rate"] == 0.0  # Can't calculate without effective_input_tokens

    def test_metrics_multiple_days(self):
        """Test metrics with data spanning multiple days"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=2),
                    base_time + timedelta(days=3),
                ],
                "cost": [25.0, 25.0, 25.0, 25.0],
                "effective_input_tokens": [1000] * 4,
                "cache_read_input_tokens": [250] * 4,
            }
        )

        result = calculate_cost_metrics(df)

        assert result["total_cost"] == 100.0
        assert round(result["daily_avg_cost"], 2) == 33.33  # 100/3 days (rounded)
        assert result["cache_hit_rate"] == 25.0  # 1000/4000*100
        assert result["avg_cost_per_response"] == 25.0

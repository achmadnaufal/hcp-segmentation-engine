"""
Tests for src/rfm_scorer.py.

Run with::

    pytest tests/test_rfm_scorer.py -q

Coverage:
- Happy-path RFM computation
- Empty DataFrame -> ValueError
- Single HCP (no quintile possible)
- All-zero metric columns
- NaN handling in dates and numerics
- Custom weights validation
- Top-N ranking and summary outputs
- Type / value error guards
- Immutability of input
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rfm_scorer import (
    DEFAULT_WEIGHTS,
    NUM_QUINTILES,
    compute_rfm_scores,
    get_top_hcps,
    summarise_rfm_segments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reference_date() -> str:
    """Stable reference date for deterministic recency calculations."""
    return "2026-04-18"


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Five-HCP DataFrame spanning the full quintile range."""
    return pd.DataFrame(
        {
            "hcp_id": ["H1", "H2", "H3", "H4", "H5"],
            "last_rx_date": [
                "2026-04-15",  # most recent
                "2026-03-20",
                "2026-02-01",
                "2025-12-10",
                "2025-09-01",  # least recent
            ],
            "rx_count_90d": [150, 90, 60, 30, 5],
            "total_rx_value_usd": [55000.0, 28000.0, 14000.0, 6000.0, 800.0],
        }
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestComputeRFM:
    def test_adds_expected_columns(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        for col in [
            "recency_days",
            "r_score",
            "f_score",
            "m_score",
            "rfm_code",
            "rfm_score",
            "rfm_segment",
        ]:
            assert col in result.columns

    def test_top_hcp_gets_highest_scores(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        h1 = result.set_index("hcp_id").loc["H1"]
        # H1 has most recent Rx, highest count, highest value -> all 5s.
        assert int(h1["r_score"]) == NUM_QUINTILES
        assert int(h1["f_score"]) == NUM_QUINTILES
        assert int(h1["m_score"]) == NUM_QUINTILES
        assert h1["rfm_code"] == "555"
        assert h1["rfm_segment"] == "Champion"

    def test_bottom_hcp_gets_lowest_scores(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        h5 = result.set_index("hcp_id").loc["H5"]
        assert int(h5["r_score"]) == 1
        assert int(h5["f_score"]) == 1
        assert int(h5["m_score"]) == 1
        assert h5["rfm_segment"] == "Lost"

    def test_recency_days_non_negative(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        assert (result["recency_days"] >= 0).all()

    def test_score_in_valid_range(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        assert (result["rfm_score"] >= 0.0).all()
        assert (result["rfm_score"] <= 100.0).all()

    def test_input_dataframe_not_mutated(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        snapshot = sample_df.copy(deep=True)
        _ = compute_rfm_scores(sample_df, reference_date=reference_date)
        pd.testing.assert_frame_equal(sample_df, snapshot)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_raises(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "hcp_id",
                "last_rx_date",
                "rx_count_90d",
                "total_rx_value_usd",
            ]
        )
        with pytest.raises(ValueError, match="empty"):
            compute_rfm_scores(empty, reference_date="2026-04-18")

    def test_single_hcp_returns_median_quintile(
        self, reference_date: str
    ) -> None:
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "last_rx_date": ["2026-04-10"],
                "rx_count_90d": [42],
                "total_rx_value_usd": [12345.0],
            }
        )
        result = compute_rfm_scores(df, reference_date=reference_date)
        median_q = NUM_QUINTILES // 2 + 1  # 3
        assert int(result.loc[0, "r_score"]) == median_q
        assert int(result.loc[0, "f_score"]) == median_q
        assert int(result.loc[0, "m_score"]) == median_q

    def test_all_zero_metrics_assigned_median_quintile(
        self, reference_date: str
    ) -> None:
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2", "H3"],
                "last_rx_date": [reference_date] * 3,
                "rx_count_90d": [0, 0, 0],
                "total_rx_value_usd": [0.0, 0.0, 0.0],
            }
        )
        result = compute_rfm_scores(df, reference_date=reference_date)
        # All scores should be the median quintile (no variance).
        assert (result["f_score"] == NUM_QUINTILES // 2 + 1).all()
        assert (result["m_score"] == NUM_QUINTILES // 2 + 1).all()
        assert (result["recency_days"] == 0).all()

    def test_nan_dates_default_to_reference(
        self, reference_date: str
    ) -> None:
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2"],
                "last_rx_date": [None, "2025-01-01"],
                "rx_count_90d": [10, 5],
                "total_rx_value_usd": [1000.0, 500.0],
            }
        )
        result = compute_rfm_scores(df, reference_date=reference_date)
        h1 = result.set_index("hcp_id").loc["H1"]
        # Missing date -> recency_days == 0 -> highest recency quintile.
        assert int(h1["recency_days"]) == 0

    def test_nan_numerics_treated_as_zero(self, reference_date: str) -> None:
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2", "H3"],
                "last_rx_date": ["2026-04-10"] * 3,
                "rx_count_90d": [np.nan, 50, 100],
                "total_rx_value_usd": [np.nan, 5000.0, 10000.0],
            }
        )
        result = compute_rfm_scores(df, reference_date=reference_date)
        h1 = result.set_index("hcp_id").loc["H1"]
        # NaN -> 0, the smallest -> lowest quintile.
        assert int(h1["f_score"]) == 1
        assert int(h1["m_score"]) == 1


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

class TestValidation:
    def test_non_dataframe_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            compute_rfm_scores("not a dataframe")  # type: ignore[arg-type]

    def test_missing_column_raises(self, reference_date: str) -> None:
        df = pd.DataFrame(
            {"hcp_id": ["H1"], "last_rx_date": ["2026-04-10"]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            compute_rfm_scores(df, reference_date=reference_date)

    def test_weights_must_sum_to_one(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        bad = {"recency": 0.5, "frequency": 0.5, "monetary": 0.5}
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_rfm_scores(
                sample_df, reference_date=reference_date, weights=bad
            )

    def test_weights_negative_rejected(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        bad = {"recency": -0.1, "frequency": 0.55, "monetary": 0.55}
        with pytest.raises(ValueError, match="non-negative"):
            compute_rfm_scores(
                sample_df, reference_date=reference_date, weights=bad
            )

    def test_weights_missing_key_rejected(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        bad = {"recency": 0.5, "frequency": 0.5}
        with pytest.raises(ValueError, match="missing required keys"):
            compute_rfm_scores(
                sample_df, reference_date=reference_date, weights=bad
            )


# ---------------------------------------------------------------------------
# Top-N and summary helpers
# ---------------------------------------------------------------------------

class TestTopAndSummary:
    def test_top_hcps_returns_n_rows_sorted(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        top3 = get_top_hcps(result, n=3)
        assert len(top3) == 3
        assert top3["rfm_score"].is_monotonic_decreasing
        assert top3.iloc[0]["hcp_id"] == "H1"

    def test_top_hcps_n_larger_than_df(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        top100 = get_top_hcps(result, n=100)
        assert len(top100) == len(result)

    def test_top_hcps_invalid_n(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        with pytest.raises(ValueError, match="positive"):
            get_top_hcps(result, n=0)
        with pytest.raises(TypeError, match="int"):
            get_top_hcps(result, n="five")  # type: ignore[arg-type]

    def test_summarise_segments(
        self, sample_df: pd.DataFrame, reference_date: str
    ) -> None:
        result = compute_rfm_scores(sample_df, reference_date=reference_date)
        summary = summarise_rfm_segments(result)
        assert {"segment", "hcp_count", "avg_rfm_score"}.issubset(
            summary.columns
        )
        assert summary["hcp_count"].sum() == len(result)
        assert summary["avg_rfm_score"].is_monotonic_decreasing

    def test_summarise_empty_raises(self) -> None:
        empty = pd.DataFrame(columns=["rfm_segment", "rfm_score"])
        with pytest.raises(ValueError, match="empty"):
            summarise_rfm_segments(empty)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_compute_rfm_is_deterministic(
    sample_df: pd.DataFrame, reference_date: str
) -> None:
    a = compute_rfm_scores(sample_df, reference_date=reference_date)
    b = compute_rfm_scores(sample_df, reference_date=reference_date)
    pd.testing.assert_frame_equal(a, b)


def test_default_weights_sum_to_one() -> None:
    assert np.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0)

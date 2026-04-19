"""
Unit tests for the potential-vs-actual bubble segmentation module.

Run with::

    pytest tests/test_potential_actual_matrix.py -v

Coverage:
- Quadrant assignment for Star / Grow / Maintain / Monitor
- Immutability (input DataFrames never mutated)
- Edge cases: empty DataFrame, single HCP, all-zero volumes, NaN values,
  duplicate HCP IDs, missing specialty, fallback actual-volume column
- Summary statistics correctness
- Top-N growth-opportunity ranking
- Input validation (type, value, required-column errors)
- Determinism across repeated runs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root when running without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.potential_actual_matrix import (
    FALLBACK_ACTUAL_COLS,
    QUADRANT_LABELS,
    compute_potential_actual_matrix,
    summarise_quadrants,
    top_growth_opportunities,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def quadrant_df() -> pd.DataFrame:
    """Four-HCP DataFrame containing exactly one record per quadrant.

    Layout (relative to median split):

    - HCP001: high potential, high actual   -> Star
    - HCP002: high potential, low actual    -> Grow
    - HCP003: low potential,  high actual   -> Maintain
    - HCP004: low potential,  low actual    -> Monitor
    """
    return pd.DataFrame(
        {
            "hcp_id": ["HCP001", "HCP002", "HCP003", "HCP004"],
            "specialty": ["Cardiology", "Oncology", "Neurology", "GP"],
            "potential_score": [95, 90, 25, 20],
            "rx_volume_last_12m": [300, 40, 280, 30],
        }
    )


@pytest.fixture()
def realistic_df() -> pd.DataFrame:
    """16-HCP realistic cohort covering all quadrants plus edge-case rows."""
    return pd.DataFrame(
        {
            "hcp_id": [f"HCP{i:03d}" for i in range(1, 17)],
            "specialty": [
                "Cardiology", "Cardiology", "Oncology", "Oncology",
                "Endocrinology", "Endocrinology", "Neurology", "Neurology",
                "GP", "GP", "Pulmonology", "Pulmonology",
                "Cardiology", "Oncology", "GP", "Pulmonology",
            ],
            "potential_score": [
                95, 90, 88, 82, 78, 72, 68, 62,
                58, 52, 48, 42, 38, 32, 28, 22,
            ],
            "rx_volume_last_12m": [
                320, 310, 50, 40, 280, 260, 30, 25,
                230, 220, 20, 18, 200, 190, 15, 10,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Core quadrant assignment
# ---------------------------------------------------------------------------

class TestQuadrantAssignment:
    """Quadrant label is correct for known (potential, actual) combos."""

    def test_all_four_quadrants_present(
        self, quadrant_df: pd.DataFrame
    ) -> None:
        """Each of the four canonical quadrants appears exactly once."""
        result = compute_potential_actual_matrix(quadrant_df)
        quadrants = result.set_index("hcp_id")["quadrant"].to_dict()
        assert quadrants["HCP001"] == "Star"
        assert quadrants["HCP002"] == "Grow"
        assert quadrants["HCP003"] == "Maintain"
        assert quadrants["HCP004"] == "Monitor"

    def test_output_contains_all_enrichment_columns(
        self, quadrant_df: pd.DataFrame
    ) -> None:
        """All six enrichment columns must be added."""
        result = compute_potential_actual_matrix(quadrant_df)
        for col in (
            "potential_norm",
            "actual_norm",
            "gap_score",
            "quadrant",
            "potential_tier",
            "actual_tier",
        ):
            assert col in result.columns, f"missing column {col}"

    def test_normalised_columns_are_in_0_100_range(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """potential_norm and actual_norm are both in [0, 100]."""
        result = compute_potential_actual_matrix(realistic_df)
        assert result["potential_norm"].min() >= 0.0
        assert result["potential_norm"].max() <= 100.0
        assert result["actual_norm"].min() >= 0.0
        assert result["actual_norm"].max() <= 100.0

    def test_gap_score_equals_potential_minus_actual(
        self, quadrant_df: pd.DataFrame
    ) -> None:
        """gap_score ~= potential_norm - actual_norm for every row."""
        result = compute_potential_actual_matrix(quadrant_df)
        diff = (result["potential_norm"] - result["actual_norm"])
        # Allow a tiny rounding tolerance because gap_score is rounded to 2dp
        # after the subtraction while the raw subtraction is not.
        for gap_val, diff_val in zip(result["gap_score"], diff):
            assert abs(gap_val - diff_val) < 0.011

    def test_quadrant_labels_are_from_known_set(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Every quadrant label is one of the canonical four."""
        result = compute_potential_actual_matrix(realistic_df)
        assert set(result["quadrant"].unique()).issubset(set(QUADRANT_LABELS))

    def test_tier_columns_contain_only_high_low(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """potential_tier and actual_tier contain only 'High' or 'Low'."""
        result = compute_potential_actual_matrix(realistic_df)
        assert set(result["potential_tier"].unique()).issubset({"High", "Low"})
        assert set(result["actual_tier"].unique()).issubset({"High", "Low"})

    @pytest.mark.parametrize(
        "split_quantile, expected_quadrant",
        [
            (0.25, "Star"),     # low cut-off -> most rows are High/High
            (0.75, "Monitor"),  # high cut-off -> most rows are Low/Low
        ],
    )
    def test_custom_split_quantile_shifts_classification(
        self,
        realistic_df: pd.DataFrame,
        split_quantile: float,
        expected_quadrant: str,
    ) -> None:
        """Non-default split quantile changes the modal quadrant."""
        result = compute_potential_actual_matrix(
            realistic_df, split_quantile=split_quantile
        )
        modal = result["quadrant"].value_counts().idxmax()
        assert modal == expected_quadrant


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Empty, single-row, zero-volume, NaN, duplicates, missing specialty."""

    def test_empty_dataframe_returns_empty_with_schema(self) -> None:
        """Empty input returns an empty DataFrame with enrichment columns."""
        empty_df = pd.DataFrame(
            {"hcp_id": [], "potential_score": [], "rx_volume_last_12m": []}
        )
        result = compute_potential_actual_matrix(empty_df)
        assert result.empty
        for col in (
            "potential_norm",
            "actual_norm",
            "gap_score",
            "quadrant",
            "potential_tier",
            "actual_tier",
        ):
            assert col in result.columns

    def test_single_hcp_classified_as_monitor(self) -> None:
        """Lone HCP cannot be 'above cohort median'; defaults to Monitor."""
        single_df = pd.DataFrame(
            {
                "hcp_id": ["HCP001"],
                "potential_score": [75],
                "rx_volume_last_12m": [200],
            }
        )
        result = compute_potential_actual_matrix(single_df)
        assert result.loc[0, "quadrant"] == "Monitor"
        assert result.loc[0, "potential_tier"] == "Low"
        assert result.loc[0, "actual_tier"] == "Low"

    def test_all_zero_volumes_produce_monitor_and_zero_gap(self) -> None:
        """Constant-zero potential and actual => every HCP is Monitor."""
        zero_df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2", "H3"],
                "potential_score": [0, 0, 0],
                "rx_volume_last_12m": [0, 0, 0],
            }
        )
        result = compute_potential_actual_matrix(zero_df)
        assert (result["quadrant"] == "Monitor").all()
        assert (result["gap_score"] == 0.0).all()

    def test_nan_in_key_columns_treated_as_zero(self) -> None:
        """NaN potential or actual is coerced to 0 before splitting."""
        nan_df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2", "H3", "H4"],
                "potential_score": [np.nan, 90, 30, 20],
                "rx_volume_last_12m": [100, np.nan, 250, 30],
            }
        )
        result = compute_potential_actual_matrix(nan_df)
        # Should not raise and should still assign a quadrant to every row.
        assert len(result) == 4
        assert result["quadrant"].notna().all()

    def test_duplicate_hcp_ids_are_preserved(self) -> None:
        """Duplicate HCP IDs do not raise and are retained row-for-row."""
        dup_df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H1", "H2"],
                "potential_score": [90, 80, 20],
                "rx_volume_last_12m": [300, 50, 25],
            }
        )
        result = compute_potential_actual_matrix(dup_df)
        assert len(result) == 3
        assert list(result["hcp_id"]) == ["H1", "H1", "H2"]

    def test_missing_specialty_column_is_ignored(self) -> None:
        """specialty is not required for matrix computation."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2"],
                "potential_score": [90, 20],
                "rx_volume_last_12m": [300, 20],
            }
        )
        result = compute_potential_actual_matrix(df)
        assert "quadrant" in result.columns
        assert len(result) == 2

    def test_fallback_actual_column_is_used(self) -> None:
        """Falls back to 'prescriptions_last_12m' when primary col absent."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2", "H3", "H4"],
                "potential_score": [90, 80, 30, 20],
                "prescriptions_last_12m": [300, 50, 280, 30],
            }
        )
        result = compute_potential_actual_matrix(df)
        quadrants = dict(zip(result["hcp_id"], result["quadrant"]))
        assert quadrants["H1"] == "Star"
        assert quadrants["H2"] == "Grow"
        assert quadrants["H3"] == "Maintain"
        assert quadrants["H4"] == "Monitor"


# ---------------------------------------------------------------------------
# Immutability & determinism
# ---------------------------------------------------------------------------

class TestImmutabilityAndDeterminism:
    """compute_potential_actual_matrix must be pure and deterministic."""

    def test_input_dataframe_is_not_mutated(
        self, quadrant_df: pd.DataFrame
    ) -> None:
        """Computing the matrix must not modify the caller's DataFrame."""
        before = quadrant_df.copy()
        _ = compute_potential_actual_matrix(quadrant_df)
        pd.testing.assert_frame_equal(quadrant_df, before)

    def test_repeated_calls_produce_identical_results(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Same inputs => identical outputs across repeated invocations."""
        a = compute_potential_actual_matrix(realistic_df)
        b = compute_potential_actual_matrix(realistic_df)
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

class TestSummariseQuadrants:
    """Shape, values, and ordering of the summary DataFrame."""

    def test_summary_row_counts_sum_to_total(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Sum of quadrant counts equals the cohort size."""
        result = compute_potential_actual_matrix(realistic_df)
        summary = summarise_quadrants(result)
        assert int(summary["hcp_count"].sum()) == len(realistic_df)

    def test_summary_percentages_sum_to_100(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """pct_of_cohort values sum (close) to 100."""
        result = compute_potential_actual_matrix(realistic_df)
        summary = summarise_quadrants(result)
        assert summary["pct_of_cohort"].sum() == pytest.approx(100.0, abs=0.1)

    def test_summary_rows_follow_canonical_order(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Summary rows are ordered Star -> Grow -> Maintain -> Monitor."""
        result = compute_potential_actual_matrix(realistic_df)
        summary = summarise_quadrants(result)
        observed = list(summary["quadrant"])
        expected = [q for q in QUADRANT_LABELS if q in observed]
        assert observed == expected

    def test_summary_contains_expected_columns(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Summary exposes every required metric column."""
        result = compute_potential_actual_matrix(realistic_df)
        summary = summarise_quadrants(result)
        for col in (
            "quadrant",
            "hcp_count",
            "avg_potential",
            "avg_actual",
            "avg_gap_score",
            "pct_of_cohort",
        ):
            assert col in summary.columns

    def test_summary_on_empty_matrix_raises(self) -> None:
        """Empty matrix DataFrame raises ValueError."""
        empty = pd.DataFrame(
            {
                "quadrant": [],
                "gap_score": [],
                "potential_norm": [],
                "actual_norm": [],
            }
        )
        with pytest.raises(ValueError, match="empty"):
            summarise_quadrants(empty)

    def test_summary_rejects_non_dataframe(self) -> None:
        """summarise_quadrants raises TypeError for non-DataFrame input."""
        with pytest.raises(TypeError):
            summarise_quadrants("not a dataframe")  # type: ignore[arg-type]

    def test_summary_raises_on_missing_columns(self) -> None:
        """summarise_quadrants raises when quadrant/gap columns absent."""
        bad = pd.DataFrame({"hcp_id": ["H1"], "quadrant": ["Star"]})
        with pytest.raises(ValueError, match="missing required columns"):
            summarise_quadrants(bad)


# ---------------------------------------------------------------------------
# Growth-opportunity ranking
# ---------------------------------------------------------------------------

class TestTopGrowthOpportunities:
    """top_growth_opportunities returns positive-gap HCPs, largest first."""

    def test_returns_only_positive_gap_rows(
        self, quadrant_df: pd.DataFrame
    ) -> None:
        """No row with gap_score <= 0 may appear in the result."""
        result = compute_potential_actual_matrix(quadrant_df)
        top = top_growth_opportunities(result, n=5)
        assert (top["gap_score"] > 0).all()

    def test_returns_at_most_n_rows(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """Cap at the requested n."""
        result = compute_potential_actual_matrix(realistic_df)
        top = top_growth_opportunities(result, n=3)
        assert len(top) <= 3

    def test_returns_rows_sorted_by_descending_gap(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """gap_score must be monotonically non-increasing in the output."""
        result = compute_potential_actual_matrix(realistic_df)
        top = top_growth_opportunities(result, n=10)
        gaps = list(top["gap_score"])
        assert gaps == sorted(gaps, reverse=True)

    def test_empty_when_no_positive_gap(self) -> None:
        """When every row has gap_score <= 0 the result is empty."""
        # All-actual-higher-than-potential scenario
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2"],
                "potential_score": [10, 20],
                "rx_volume_last_12m": [500, 600],
            }
        )
        result = compute_potential_actual_matrix(df)
        top = top_growth_opportunities(result, n=5)
        assert top.empty

    def test_raises_on_non_positive_n(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """n must be a positive integer."""
        result = compute_potential_actual_matrix(realistic_df)
        with pytest.raises(ValueError, match="positive integer"):
            top_growth_opportunities(result, n=0)

    def test_raises_on_non_int_n(
        self, realistic_df: pd.DataFrame
    ) -> None:
        """n must be of type int (not float)."""
        result = compute_potential_actual_matrix(realistic_df)
        with pytest.raises(TypeError):
            top_growth_opportunities(result, n=3.0)  # type: ignore[arg-type]

    def test_raises_on_missing_gap_score(self) -> None:
        """Missing 'gap_score' column raises ValueError."""
        bad = pd.DataFrame({"hcp_id": ["H1"], "quadrant": ["Star"]})
        with pytest.raises(ValueError, match="gap_score"):
            top_growth_opportunities(bad, n=1)

    def test_raises_on_non_dataframe(self) -> None:
        """Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError):
            top_growth_opportunities([1, 2, 3], n=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Input-validation guards
# ---------------------------------------------------------------------------

class TestValidation:
    """compute_potential_actual_matrix guards against bad inputs."""

    def test_raises_on_non_dataframe(self) -> None:
        """Non-DataFrame input raises TypeError with clear message."""
        with pytest.raises(TypeError, match="DataFrame"):
            compute_potential_actual_matrix({"hcp_id": [1]})  # type: ignore[arg-type]

    def test_raises_on_missing_potential_column(self) -> None:
        """Missing potential column triggers KeyError."""
        df = pd.DataFrame(
            {"hcp_id": ["H1"], "rx_volume_last_12m": [100]}
        )
        with pytest.raises(KeyError, match="potential_score"):
            compute_potential_actual_matrix(df)

    def test_raises_on_missing_actual_column_and_no_fallback(self) -> None:
        """Missing actual column (no fallback) raises KeyError."""
        df = pd.DataFrame(
            {"hcp_id": ["H1"], "potential_score": [50]}
        )
        with pytest.raises(KeyError, match="actual"):
            compute_potential_actual_matrix(df)

    @pytest.mark.parametrize("bad_q", [-0.1, 0.0, 1.0, 1.5])
    def test_raises_on_invalid_split_quantile(
        self, quadrant_df: pd.DataFrame, bad_q: float
    ) -> None:
        """split_quantile must lie strictly in (0, 1)."""
        with pytest.raises(ValueError, match="split_quantile"):
            compute_potential_actual_matrix(quadrant_df, split_quantile=bad_q)


# ---------------------------------------------------------------------------
# Fallback-column constant sanity
# ---------------------------------------------------------------------------

def test_fallback_columns_include_both_schemas() -> None:
    """The fallback list must include the two canonical schemas."""
    assert "rx_volume_last_12m" in FALLBACK_ACTUAL_COLS
    assert "prescriptions_last_12m" in FALLBACK_ACTUAL_COLS

"""
Unit tests for :mod:`src.calling_plan_allocator`.

Covers:

- ``calculate_priority_score`` happy path, edge cases, and error handling.
- ``allocate_calls`` budget conservation, per-HCP caps, zero budget,
  single-HCP, identical-priority, missing segments, NaN composite scores,
  and immutability.
- ``summarise_allocation`` structure and aggregate correctness.

Run with::

    pytest tests/test_calling_plan_allocator.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calling_plan_allocator import (
    DEFAULT_MAX_CALLS_PER_HCP,
    DEFAULT_SEGMENT_WEIGHTS,
    DEFAULT_TARGET_CALLS_PER_SEGMENT,
    allocate_calls,
    calculate_priority_score,
    summarise_allocation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def segmented_df() -> pd.DataFrame:
    """Return a 10-row HCP DataFrame covering all six default segments."""
    return pd.DataFrame(
        {
            "hcp_id": [f"HCP{i:03d}" for i in range(1, 11)],
            "computed_segment": [
                "High-Value KOL",
                "High-Value KOL",
                "Growth Target",
                "Growth Target",
                "Digital Adopter",
                "Digital Adopter",
                "Standard",
                "Standard",
                "Low Activity",
                "Dormant",
            ],
            "composite_score": [92.0, 85.0, 70.0, 65.0, 55.0, 50.0, 40.0, 35.0, 15.0, 3.0],
        }
    )


@pytest.fixture()
def single_hcp_df() -> pd.DataFrame:
    """Return a one-row HCP DataFrame."""
    return pd.DataFrame(
        {
            "hcp_id": ["HCP001"],
            "computed_segment": ["Growth Target"],
            "composite_score": [72.5],
        }
    )


# ---------------------------------------------------------------------------
# calculate_priority_score
# ---------------------------------------------------------------------------

class TestCalculatePriorityScore:
    """Tests for :func:`calculate_priority_score`."""

    def test_adds_priority_score_column(self, segmented_df: pd.DataFrame) -> None:
        """Result must contain a ``priority_score`` column."""
        result = calculate_priority_score(segmented_df)
        assert "priority_score" in result.columns

    def test_priority_score_in_zero_to_hundred(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """All priority scores must lie in ``[0, 100]``."""
        result = calculate_priority_score(segmented_df)
        assert result["priority_score"].between(0.0, 100.0).all()

    def test_kol_scores_higher_than_dormant(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """A High-Value KOL must receive a higher priority than a Dormant HCP."""
        result = calculate_priority_score(segmented_df)
        kol_idx = result.index[result["computed_segment"] == "High-Value KOL"][0]
        dormant_idx = result.index[result["computed_segment"] == "Dormant"][0]
        assert (
            result.at[kol_idx, "priority_score"]
            > result.at[dormant_idx, "priority_score"]
        )

    def test_nan_composite_treated_as_zero(self) -> None:
        """NaN composite scores must not raise and must be treated as zero."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "computed_segment": ["Standard"],
                "composite_score": [np.nan],
            }
        )
        result = calculate_priority_score(df, composite_blend=1.0)
        assert result["priority_score"].iloc[0] == pytest.approx(0.0)

    def test_unknown_segment_gets_zero_weight(self) -> None:
        """An unknown segment label must receive segment weight 0.0."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "computed_segment": ["Unknown Segment"],
                "composite_score": [50.0],
            }
        )
        result = calculate_priority_score(df, composite_blend=0.0)
        # composite_blend=0 -> score comes entirely from segment weight (0)
        assert result["priority_score"].iloc[0] == pytest.approx(0.0)

    def test_custom_segment_weights_override_defaults(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """Custom segment weights must change the priority score."""
        custom = {seg: 0.0 for seg in DEFAULT_SEGMENT_WEIGHTS}
        custom["Dormant"] = 1.0
        result = calculate_priority_score(
            segmented_df, segment_weights=custom, composite_blend=0.0
        )
        dormant_scores = result.loc[
            result["computed_segment"] == "Dormant", "priority_score"
        ]
        assert (dormant_scores == 100.0).all()

    def test_immutability(self, segmented_df: pd.DataFrame) -> None:
        """Input DataFrame must not be mutated."""
        original_cols = list(segmented_df.columns)
        calculate_priority_score(segmented_df)
        assert list(segmented_df.columns) == original_cols

    def test_raises_on_non_dataframe(self) -> None:
        """A non-DataFrame input must raise TypeError."""
        with pytest.raises(TypeError):
            calculate_priority_score([1, 2, 3])  # type: ignore[arg-type]

    def test_raises_on_missing_columns(self) -> None:
        """A DataFrame missing required columns must raise ValueError."""
        with pytest.raises(ValueError, match="missing required columns"):
            calculate_priority_score(pd.DataFrame({"hcp_id": ["H1"]}))

    def test_raises_on_invalid_blend(self, segmented_df: pd.DataFrame) -> None:
        """composite_blend outside [0, 1] must raise ValueError."""
        with pytest.raises(ValueError, match="composite_blend"):
            calculate_priority_score(segmented_df, composite_blend=1.5)

    def test_raises_on_invalid_segment_weight(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """Segment weights outside [0, 1] must raise ValueError."""
        bad_weights = {"High-Value KOL": 1.5}
        with pytest.raises(ValueError, match="segment_weights"):
            calculate_priority_score(segmented_df, segment_weights=bad_weights)


# ---------------------------------------------------------------------------
# allocate_calls
# ---------------------------------------------------------------------------

class TestAllocateCalls:
    """Tests for :func:`allocate_calls`."""

    def test_allocation_conserves_total_budget(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """Sum of allocated calls must equal the requested budget."""
        budget = 200
        result = allocate_calls(segmented_df, total_calls_budget=budget)
        assert int(result["allocated_calls"].sum()) == budget

    def test_zero_budget_produces_zero_calls(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """A zero budget must result in zero calls allocated to every HCP."""
        result = allocate_calls(segmented_df, total_calls_budget=0)
        assert (result["allocated_calls"] == 0).all()

    def test_min_calls_per_hcp_honoured(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """No HCP may receive fewer than min_calls_per_hcp."""
        min_calls = 2
        total = 100
        result = allocate_calls(
            segmented_df,
            total_calls_budget=total,
            min_calls_per_hcp=min_calls,
        )
        assert (result["allocated_calls"] >= min_calls).all()

    def test_max_calls_per_hcp_honoured(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """No HCP may receive more than max_calls_per_hcp."""
        max_calls = 20
        result = allocate_calls(
            segmented_df,
            total_calls_budget=2000,  # intentionally larger than capacity
            max_calls_per_hcp=max_calls,
        )
        assert (result["allocated_calls"] <= max_calls).all()

    def test_kol_receives_more_calls_than_dormant(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """High-Value KOLs must receive more calls than Dormant HCPs."""
        result = allocate_calls(segmented_df, total_calls_budget=150)
        kol_calls = result.loc[
            result["computed_segment"] == "High-Value KOL", "allocated_calls"
        ].max()
        dormant_calls = result.loc[
            result["computed_segment"] == "Dormant", "allocated_calls"
        ].max()
        assert kol_calls > dormant_calls

    def test_single_hcp_gets_entire_budget(
        self, single_hcp_df: pd.DataFrame
    ) -> None:
        """With one HCP and budget <= max cap, that HCP gets the full budget."""
        result = allocate_calls(
            single_hcp_df,
            total_calls_budget=20,
            max_calls_per_hcp=50,
        )
        assert result["allocated_calls"].iloc[0] == 20

    def test_single_hcp_capped_at_max(
        self, single_hcp_df: pd.DataFrame
    ) -> None:
        """A single HCP cannot receive more than max_calls_per_hcp."""
        result = allocate_calls(
            single_hcp_df,
            total_calls_budget=1000,
            max_calls_per_hcp=30,
        )
        assert result["allocated_calls"].iloc[0] == 30

    def test_empty_dataframe_returns_empty_with_columns(self) -> None:
        """An empty DataFrame must yield an empty result with expected columns."""
        df = pd.DataFrame(columns=["hcp_id", "computed_segment", "composite_score"])
        result = allocate_calls(df, total_calls_budget=100)
        assert result.empty
        assert "allocated_calls" in result.columns
        assert "priority_score" in result.columns

    def test_identical_priorities_distribute_evenly(self) -> None:
        """Identical priorities must produce a roughly-even distribution."""
        df = pd.DataFrame(
            {
                "hcp_id": [f"H{i}" for i in range(4)],
                "computed_segment": ["Standard"] * 4,
                "composite_score": [50.0] * 4,
            }
        )
        result = allocate_calls(df, total_calls_budget=20)
        spread = int(result["allocated_calls"].max() - result["allocated_calls"].min())
        # Allow at most one call of drift across HCPs due to integer rounding.
        assert spread <= 1

    def test_nan_composite_scores_do_not_crash(self) -> None:
        """NaN composite scores must not raise and must produce valid ints."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2"],
                "computed_segment": ["Growth Target", "Standard"],
                "composite_score": [np.nan, 50.0],
            }
        )
        result = allocate_calls(df, total_calls_budget=30)
        assert result["allocated_calls"].dtype.kind == "i"
        assert int(result["allocated_calls"].sum()) == 30

    def test_immutability(self, segmented_df: pd.DataFrame) -> None:
        """allocate_calls() must not mutate the input DataFrame."""
        original_cols = list(segmented_df.columns)
        allocate_calls(segmented_df, total_calls_budget=100)
        assert list(segmented_df.columns) == original_cols

    def test_allocation_is_deterministic(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """Running the allocation twice with same inputs must give same output."""
        r1 = allocate_calls(segmented_df, total_calls_budget=120)
        r2 = allocate_calls(segmented_df, total_calls_budget=120)
        assert (r1["allocated_calls"].values == r2["allocated_calls"].values).all()

    def test_negative_budget_raises(self, segmented_df: pd.DataFrame) -> None:
        """A negative budget must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            allocate_calls(segmented_df, total_calls_budget=-10)

    def test_non_int_budget_raises(self, segmented_df: pd.DataFrame) -> None:
        """A non-integer budget must raise TypeError."""
        with pytest.raises(TypeError):
            allocate_calls(segmented_df, total_calls_budget=3.5)  # type: ignore[arg-type]

    def test_min_exceeds_max_raises(self, segmented_df: pd.DataFrame) -> None:
        """min_calls_per_hcp > max_calls_per_hcp must raise ValueError."""
        with pytest.raises(ValueError, match="max_calls_per_hcp"):
            allocate_calls(
                segmented_df,
                total_calls_budget=100,
                min_calls_per_hcp=10,
                max_calls_per_hcp=5,
            )

    def test_non_dataframe_raises(self) -> None:
        """A non-DataFrame input must raise TypeError."""
        with pytest.raises(TypeError):
            allocate_calls(None, total_calls_budget=50)  # type: ignore[arg-type]

    def test_budget_exceeds_capacity_saturates(self) -> None:
        """When budget > per-HCP caps * HCPs, allocation saturates at cap."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1", "H2"],
                "computed_segment": ["High-Value KOL", "High-Value KOL"],
                "composite_score": [90.0, 80.0],
            }
        )
        result = allocate_calls(
            df,
            total_calls_budget=10_000,
            max_calls_per_hcp=5,
        )
        assert (result["allocated_calls"] == 5).all()


# ---------------------------------------------------------------------------
# summarise_allocation
# ---------------------------------------------------------------------------

class TestSummariseAllocation:
    """Tests for :func:`summarise_allocation`."""

    def test_summary_columns(self, segmented_df: pd.DataFrame) -> None:
        """Summary must contain the documented columns."""
        allocation = allocate_calls(segmented_df, total_calls_budget=150)
        summary = summarise_allocation(allocation)
        expected = {
            "segment",
            "hcp_count",
            "total_calls",
            "avg_calls_per_hcp",
            "avg_priority",
            "pct_of_budget",
        }
        assert expected.issubset(set(summary.columns))

    def test_summary_total_calls_match_allocation(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """Sum of per-segment total_calls must equal the overall budget."""
        budget = 180
        allocation = allocate_calls(segmented_df, total_calls_budget=budget)
        summary = summarise_allocation(allocation)
        assert int(summary["total_calls"].sum()) == budget

    def test_summary_pct_of_budget_sums_to_100(
        self, segmented_df: pd.DataFrame
    ) -> None:
        """pct_of_budget values must sum to ~100%."""
        allocation = allocate_calls(segmented_df, total_calls_budget=150)
        summary = summarise_allocation(allocation)
        assert summary["pct_of_budget"].sum() == pytest.approx(100.0, abs=0.5)

    def test_raises_on_empty(self) -> None:
        """Empty input must raise ValueError."""
        df = pd.DataFrame(
            columns=["computed_segment", "allocated_calls", "priority_score"]
        )
        with pytest.raises(ValueError, match="empty"):
            summarise_allocation(df)

    def test_raises_on_missing_columns(self) -> None:
        """Missing columns must raise ValueError."""
        df = pd.DataFrame(
            {"segment": ["A"], "calls": [1]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            summarise_allocation(df)

    def test_raises_on_non_dataframe(self) -> None:
        """Non-DataFrame input must raise TypeError."""
        with pytest.raises(TypeError):
            summarise_allocation({"a": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default constants sanity
# ---------------------------------------------------------------------------

class TestDefaultConstants:
    """Sanity checks for module-level default constants."""

    def test_default_segment_weights_in_unit_range(self) -> None:
        """Every default segment weight must lie in ``[0, 1]``."""
        for seg, w in DEFAULT_SEGMENT_WEIGHTS.items():
            assert 0.0 <= w <= 1.0, f"{seg} weight {w} out of range"

    def test_default_targets_non_negative(self) -> None:
        """Every default target calls-per-segment must be non-negative."""
        for seg, t in DEFAULT_TARGET_CALLS_PER_SEGMENT.items():
            assert t >= 0, f"{seg} target {t} must be non-negative"

    def test_default_max_calls_positive(self) -> None:
        """Default per-HCP max cap must be positive."""
        assert DEFAULT_MAX_CALLS_PER_HCP > 0

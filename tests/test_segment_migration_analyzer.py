"""
Tests for src/segment_migration_analyzer.py.

Run with::

    pytest tests/test_segment_migration_analyzer.py -q

Coverage targets:
- Happy-path migration table, matrix, and summary
- Empty / single-HCP inputs
- Tied churn-risk scores
- Fully stable cohort
- Fully churned cohort
- Unknown segment names
- normalise=True matrix
- Determinism (same output on repeated calls)
- Type / value error guards
- Parametrised direction cases
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segment_migration_analyzer import (
    SEGMENT_ORDER,
    build_migration_matrix,
    compute_migration_table,
    summarise_migrations,
    _segment_rank,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_before() -> pd.DataFrame:
    """Three-HCP DataFrame for period A."""
    return pd.DataFrame(
        {
            "hcp_id": ["H1", "H2", "H3"],
            "computed_segment": ["Standard", "Growth Target", "Dormant"],
        }
    )


@pytest.fixture()
def simple_after() -> pd.DataFrame:
    """Three-HCP DataFrame for period B."""
    return pd.DataFrame(
        {
            "hcp_id": ["H1", "H2", "H3"],
            "computed_segment": ["Growth Target", "Standard", "Low Activity"],
        }
    )


@pytest.fixture()
def migration_table(simple_before, simple_after) -> pd.DataFrame:
    return compute_migration_table(simple_before, simple_after)


# ---------------------------------------------------------------------------
# 1. _segment_rank helper
# ---------------------------------------------------------------------------

class TestSegmentRank:
    def test_known_segments_ordered(self):
        ranks = [_segment_rank(s) for s in SEGMENT_ORDER]
        assert ranks == list(range(len(SEGMENT_ORDER)))

    def test_unknown_segment_returns_minus_one(self):
        assert _segment_rank("Alien Segment") == -1


# ---------------------------------------------------------------------------
# 2. compute_migration_table — happy path
# ---------------------------------------------------------------------------

class TestComputeMigrationTable:
    def test_returns_dataframe(self, migration_table):
        assert isinstance(migration_table, pd.DataFrame)

    def test_expected_columns_present(self, migration_table):
        expected = {
            "hcp_id", "segment_before", "segment_after",
            "rank_before", "rank_after", "rank_delta",
            "direction", "is_churned", "churn_risk_score",
        }
        assert expected.issubset(set(migration_table.columns))

    def test_row_count_equals_matched_hcps(self, simple_before, simple_after):
        result = compute_migration_table(simple_before, simple_after)
        assert len(result) == 3

    def test_upgrade_detected(self, migration_table):
        h1 = migration_table[migration_table["hcp_id"] == "H1"].iloc[0]
        assert h1["direction"] == "upgrade"
        assert h1["rank_delta"] > 0

    def test_downgrade_detected(self, migration_table):
        h2 = migration_table[migration_table["hcp_id"] == "H2"].iloc[0]
        assert h2["direction"] == "downgrade"
        assert h2["rank_delta"] < 0

    def test_stable_hcp_from_dormant(self, migration_table):
        # H3 goes Dormant -> Low Activity (upgrade, rank -1 -> 1)
        h3 = migration_table[migration_table["hcp_id"] == "H3"].iloc[0]
        assert h3["direction"] == "upgrade"

    def test_churn_risk_zero_for_upgrade(self, migration_table):
        h1 = migration_table[migration_table["hcp_id"] == "H1"].iloc[0]
        assert h1["churn_risk_score"] == 0.0

    def test_churn_risk_positive_for_downgrade(self, migration_table):
        h2 = migration_table[migration_table["hcp_id"] == "H2"].iloc[0]
        assert h2["churn_risk_score"] > 0.0

    def test_is_churned_flag_when_dormant_after(self):
        before = pd.DataFrame({
            "hcp_id": ["X1"],
            "computed_segment": ["Standard"],
        })
        after = pd.DataFrame({
            "hcp_id": ["X1"],
            "computed_segment": ["Dormant"],
        })
        result = compute_migration_table(before, after)
        assert result.iloc[0]["is_churned"] is True or result.iloc[0]["is_churned"] == True

    def test_only_matched_hcps_included(self):
        before = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Standard", "Dormant"],
        })
        after = pd.DataFrame({
            "hcp_id": ["A", "C"],          # C is new, B dropped out
            "computed_segment": ["Standard", "Standard"],
        })
        result = compute_migration_table(before, after)
        assert len(result) == 1
        assert result.iloc[0]["hcp_id"] == "A"

    def test_determinism(self, simple_before, simple_after):
        r1 = compute_migration_table(simple_before, simple_after)
        r2 = compute_migration_table(simple_before, simple_after)
        pd.testing.assert_frame_equal(r1, r2)

    def test_single_hcp(self):
        before = pd.DataFrame({
            "hcp_id": ["S1"],
            "computed_segment": ["Low Activity"],
        })
        after = pd.DataFrame({
            "hcp_id": ["S1"],
            "computed_segment": ["Standard"],
        })
        result = compute_migration_table(before, after)
        assert len(result) == 1
        assert result.iloc[0]["direction"] == "upgrade"

    def test_input_dataframes_not_mutated(self, simple_before, simple_after):
        cols_before = list(simple_before.columns)
        cols_after = list(simple_after.columns)
        compute_migration_table(simple_before, simple_after)
        assert list(simple_before.columns) == cols_before
        assert list(simple_after.columns) == cols_after


# ---------------------------------------------------------------------------
# 3. compute_migration_table — error paths
# ---------------------------------------------------------------------------

class TestComputeMigrationTableErrors:
    def test_raises_type_error_for_non_df(self):
        with pytest.raises(TypeError):
            compute_migration_table("not a df", pd.DataFrame({"hcp_id": ["x"], "computed_segment": ["Dormant"]}))

    def test_raises_value_error_for_empty_df(self):
        with pytest.raises(ValueError):
            compute_migration_table(
                pd.DataFrame(),
                pd.DataFrame({"hcp_id": ["x"], "computed_segment": ["Dormant"]}),
            )

    def test_raises_value_error_for_missing_column(self):
        bad = pd.DataFrame({"hcp_id": ["x"]})   # missing segment col
        good = pd.DataFrame({"hcp_id": ["x"], "computed_segment": ["Dormant"]})
        with pytest.raises(ValueError):
            compute_migration_table(bad, good)


# ---------------------------------------------------------------------------
# 4. build_migration_matrix
# ---------------------------------------------------------------------------

class TestBuildMigrationMatrix:
    def test_returns_dataframe(self, migration_table):
        mx = build_migration_matrix(migration_table)
        assert isinstance(mx, pd.DataFrame)

    def test_matrix_cell_counts_correct(self):
        before = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Standard", "Standard"],
        })
        after = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Growth Target", "Standard"],
        })
        tbl = compute_migration_table(before, after)
        mx = build_migration_matrix(tbl)
        assert mx.loc["Standard", "Growth Target"] == 1
        assert mx.loc["Standard", "Standard"] == 1

    def test_normalise_rows_sum_to_one(self, migration_table):
        mx = build_migration_matrix(migration_table, normalise=True)
        row_sums = mx.sum(axis=1)
        # Each row that has at least one non-zero entry sums to ~1
        for val in row_sums:
            assert abs(val - 1.0) < 1e-6 or abs(val) < 1e-6

    def test_raises_type_error_for_non_df(self):
        with pytest.raises(TypeError):
            build_migration_matrix(42)

    def test_raises_value_error_for_empty_df(self):
        with pytest.raises(ValueError):
            build_migration_matrix(pd.DataFrame())


# ---------------------------------------------------------------------------
# 5. summarise_migrations
# ---------------------------------------------------------------------------

class TestSummariseMigrations:
    def test_returns_dict(self, migration_table):
        stats = summarise_migrations(migration_table)
        assert isinstance(stats, dict)

    def test_total_hcps_correct(self, migration_table):
        stats = summarise_migrations(migration_table)
        assert stats["total_hcps"] == 3

    def test_upgrade_count(self, migration_table):
        stats = summarise_migrations(migration_table)
        # H1 upgrades, H3 upgrades (Dormant->Low Activity via rank -1->1)
        assert stats["upgraded"] == 2

    def test_downgrade_count(self, migration_table):
        stats = summarise_migrations(migration_table)
        assert stats["downgraded"] == 1

    def test_churned_count_zero_when_no_dormant_after(self, migration_table):
        # None of H1/H2/H3 end in Dormant
        stats = summarise_migrations(migration_table)
        assert stats["churned"] == 0

    def test_fully_stable_cohort(self):
        before = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Standard", "Low Activity"],
        })
        after = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Standard", "Low Activity"],
        })
        tbl = compute_migration_table(before, after)
        stats = summarise_migrations(tbl)
        assert stats["stable"] == 2
        assert stats["upgraded"] == 0
        assert stats["downgraded"] == 0

    def test_fully_churned_cohort(self):
        before = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Standard", "Growth Target"],
        })
        after = pd.DataFrame({
            "hcp_id": ["A", "B"],
            "computed_segment": ["Dormant", "Dormant"],
        })
        tbl = compute_migration_table(before, after)
        stats = summarise_migrations(tbl)
        assert stats["churned"] == 2
        assert stats["pct_churned"] == 100.0

    def test_top_churn_risk_hcps_list_type(self, migration_table):
        stats = summarise_migrations(migration_table)
        assert isinstance(stats["top_churn_risk_hcps"], list)

    def test_avg_churn_risk_is_float(self, migration_table):
        stats = summarise_migrations(migration_table)
        assert isinstance(stats["avg_churn_risk"], float)

    def test_raises_type_error_for_non_df(self):
        with pytest.raises(TypeError):
            summarise_migrations("not a df")

    def test_raises_value_error_for_empty_df(self):
        with pytest.raises(ValueError):
            summarise_migrations(pd.DataFrame())


# ---------------------------------------------------------------------------
# 6. Parametrised direction cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seg_before, seg_after, expected_direction",
    [
        ("Dormant", "High-Value KOL", "upgrade"),
        ("High-Value KOL", "Dormant", "downgrade"),
        ("Standard", "Standard", "stable"),
        ("Low Activity", "Growth Target", "upgrade"),
        ("Growth Target", "Low Activity", "downgrade"),
    ],
)
def test_direction_parametrised(seg_before: str, seg_after: str, expected_direction: str):
    before = pd.DataFrame({"hcp_id": ["P1"], "computed_segment": [seg_before]})
    after = pd.DataFrame({"hcp_id": ["P1"], "computed_segment": [seg_after]})
    result = compute_migration_table(before, after)
    assert result.iloc[0]["direction"] == expected_direction

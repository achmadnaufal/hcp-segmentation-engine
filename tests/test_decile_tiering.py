"""
Unit tests for the decile-based HCP tiering module.

Run with::

    pytest tests/test_decile_tiering.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root when running without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decile_tiering import (  # noqa: E402
    DECILE_TO_LETTER,
    DEFAULT_RX_COLUMN,
    DecileReport,
    DecileTieringEngine,
    LETTER_ORDER,
    LetterSummary,
    assign_decile_tiers,
    letter_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine() -> DecileTieringEngine:
    """Return a default DecileTieringEngine."""
    return DecileTieringEngine()


@pytest.fixture()
def ten_hcp_df() -> pd.DataFrame:
    """Return 10 HCPs with strictly decreasing Rx (one per decile)."""
    return pd.DataFrame(
        {
            "hcp_id": [f"HCP{i:03d}" for i in range(1, 11)],
            "specialty": [
                "Cardiology", "Oncology", "Diabetes", "Neurology", "Primary Care",
                "Rheumatology", "Endocrinology", "Pulmonology", "Psychiatry", "Dermatology",
            ],
            "prescriptions_last_12m": [500, 450, 400, 350, 300, 250, 200, 150, 100, 50],
        }
    )


@pytest.fixture()
def twenty_hcp_df() -> pd.DataFrame:
    """Return 20 HCPs spanning the full Rx range."""
    rng = np.random.default_rng(seed=42)
    rx_values = sorted(rng.integers(low=10, high=600, size=20).tolist(), reverse=True)
    return pd.DataFrame(
        {
            "hcp_id": [f"HCP{i:03d}" for i in range(1, 21)],
            "specialty": ["Cardiology"] * 5
            + ["Oncology"] * 5
            + ["Diabetes"] * 5
            + ["Neurology"] * 5,
            "prescriptions_last_12m": rx_values,
        }
    )


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    """Tests for engine construction and input validation."""

    def test_default_construction(self) -> None:
        """Engine constructs without arguments using sensible defaults."""
        engine = DecileTieringEngine()
        assert engine is not None

    def test_custom_column_construction(self) -> None:
        """Engine accepts custom Rx and ID column names."""
        engine = DecileTieringEngine(rx_column="nrx", id_column="npi")
        # Smoke-check: no exception on construction.
        assert engine is not None

    def test_empty_rx_column_rejected(self) -> None:
        """Empty rx_column string must raise ValueError."""
        with pytest.raises(ValueError, match="rx_column"):
            DecileTieringEngine(rx_column="")

    def test_empty_id_column_rejected(self) -> None:
        """Empty id_column string must raise ValueError."""
        with pytest.raises(ValueError, match="id_column"):
            DecileTieringEngine(id_column="")

    def test_validate_empty_dataframe_raises(self, engine: DecileTieringEngine) -> None:
        """Empty DataFrame fails validation."""
        with pytest.raises(ValueError, match="empty"):
            engine.validate(pd.DataFrame())

    def test_validate_missing_rx_column_raises(
        self, engine: DecileTieringEngine
    ) -> None:
        """Missing Rx column raises a descriptive ValueError."""
        df = pd.DataFrame({"hcp_id": ["A"], "other": [1]})
        with pytest.raises(ValueError, match="Rx column"):
            engine.validate(df)

    def test_validate_missing_id_column_raises(
        self, engine: DecileTieringEngine
    ) -> None:
        """Missing ID column raises a descriptive ValueError."""
        df = pd.DataFrame({"prescriptions_last_12m": [10, 20]})
        with pytest.raises(ValueError, match="ID column"):
            engine.validate(df)

    def test_validate_non_numeric_rx_raises(
        self, engine: DecileTieringEngine
    ) -> None:
        """Non-numeric Rx column raises a ValueError."""
        df = pd.DataFrame(
            {"hcp_id": ["A", "B"], "prescriptions_last_12m": ["high", "low"]}
        )
        with pytest.raises(ValueError, match="numeric"):
            engine.validate(df)


# ---------------------------------------------------------------------------
# 2. Decile assignment
# ---------------------------------------------------------------------------

class TestDecileAssignment:
    """Tests for DecileTieringEngine.assign_deciles()."""

    def test_decile_column_created(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """assign_deciles() must add an 'rx_decile' column."""
        result = engine.assign_deciles(ten_hcp_df)
        assert "rx_decile" in result.columns

    def test_decile_values_in_range(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """All decile values must lie in [1, 10]."""
        result = engine.assign_deciles(twenty_hcp_df)
        assert result["rx_decile"].between(1, 10).all()

    def test_ten_hcps_cover_all_deciles(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """Ten HCPs with unique Rx values must occupy deciles 1..10."""
        result = engine.assign_deciles(ten_hcp_df)
        assert sorted(result["rx_decile"].tolist()) == list(range(1, 11))

    def test_top_rx_assigned_decile_1(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """The highest-prescribing HCP must land in decile 1."""
        result = engine.assign_deciles(ten_hcp_df)
        top_idx = ten_hcp_df["prescriptions_last_12m"].idxmax()
        assert result.loc[top_idx, "rx_decile"] == 1

    def test_lowest_rx_assigned_decile_10(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """The lowest-prescribing HCP must land in decile 10."""
        result = engine.assign_deciles(ten_hcp_df)
        bottom_idx = ten_hcp_df["prescriptions_last_12m"].idxmin()
        assert result.loc[bottom_idx, "rx_decile"] == 10

    def test_single_hcp_decile_assignment(self, engine: DecileTieringEngine) -> None:
        """A single HCP must receive a valid decile (1)."""
        df = pd.DataFrame({"hcp_id": ["A"], "prescriptions_last_12m": [42]})
        result = engine.assign_deciles(df)
        assert result["rx_decile"].iloc[0] == 1

    def test_duplicate_rx_values_still_unique_deciles(
        self, engine: DecileTieringEngine
    ) -> None:
        """Duplicate Rx values are broken by input order so each HCP gets a decile."""
        df = pd.DataFrame(
            {
                "hcp_id": [f"H{i}" for i in range(10)],
                "prescriptions_last_12m": [100] * 10,
            }
        )
        result = engine.assign_deciles(df)
        assert sorted(result["rx_decile"].tolist()) == list(range(1, 11))

    def test_assign_deciles_does_not_mutate_input(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """assign_deciles() returns a new DataFrame without mutating input."""
        original_cols = list(ten_hcp_df.columns)
        engine.assign_deciles(ten_hcp_df)
        assert list(ten_hcp_df.columns) == original_cols

    def test_missing_rx_filled_as_zero(self, engine: DecileTieringEngine) -> None:
        """NaN Rx values are treated as 0 and end up in the lowest decile."""
        df = pd.DataFrame(
            {
                "hcp_id": [f"H{i}" for i in range(10)],
                "prescriptions_last_12m": [500, 400, 300, 200, 100, 80, 60, 40, 20, np.nan],
            }
        )
        result = engine.assign_deciles(df)
        nan_row = result.iloc[9]
        assert nan_row["rx_decile"] == 10


# ---------------------------------------------------------------------------
# 3. Letter tier assignment
# ---------------------------------------------------------------------------

class TestLetterTiers:
    """Tests for DecileTieringEngine.assign_letter_tiers()."""

    def test_letter_column_created(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """assign_letter_tiers() must add a 'letter_tier' column."""
        with_deciles = engine.assign_deciles(ten_hcp_df)
        result = engine.assign_letter_tiers(with_deciles)
        assert "letter_tier" in result.columns

    def test_letter_tiers_match_expected_mapping(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """With 10 uniquely-ordered HCPs the letter sequence is A,A,B,B,C,C,D,D,E,E."""
        with_deciles = engine.assign_deciles(ten_hcp_df)
        result = engine.assign_letter_tiers(with_deciles)
        expected = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
        assert result["letter_tier"].tolist() == expected

    def test_letter_tiers_use_only_abcde(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """Letter tier values come only from {A, B, C, D, E}."""
        with_deciles = engine.assign_deciles(twenty_hcp_df)
        result = engine.assign_letter_tiers(with_deciles)
        assert set(result["letter_tier"]).issubset(set(LETTER_ORDER))

    def test_assign_letter_tiers_without_decile_raises(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """Running letter-tier assignment before decile assignment raises."""
        with pytest.raises(KeyError, match="rx_decile"):
            engine.assign_letter_tiers(ten_hcp_df)

    def test_decile_to_letter_covers_all_ten(self) -> None:
        """Every decile 1..10 must have a letter mapping."""
        for d in range(1, 11):
            assert d in DECILE_TO_LETTER
            assert DECILE_TO_LETTER[d] in LETTER_ORDER


# ---------------------------------------------------------------------------
# 4. Summaries and reporting
# ---------------------------------------------------------------------------

class TestSummaries:
    """Tests for summarise() and DecileReport."""

    def test_summarise_returns_five_letters(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """summarise() always returns exactly 5 entries (A..E)."""
        report = engine.run(ten_hcp_df)
        assert len(report.summaries) == 5
        assert [s.letter for s in report.summaries] == LETTER_ORDER

    def test_rx_share_sums_to_100(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """Rx shares across all letters must sum to ~100 percent."""
        report = engine.run(twenty_hcp_df)
        total_share = sum(s.rx_share_pct for s in report.summaries)
        assert abs(total_share - 100.0) < 0.5

    def test_hcp_count_sums_to_total(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """Per-letter HCP counts must sum to the total row count."""
        report = engine.run(twenty_hcp_df)
        assert sum(s.hcp_count for s in report.summaries) == len(twenty_hcp_df)

    def test_top_letter_has_highest_mean_rx(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """Tier A's mean Rx must be the highest among all letter tiers."""
        report = engine.run(twenty_hcp_df)
        by_letter = {s.letter: s.mean_rx for s in report.summaries if s.hcp_count > 0}
        assert by_letter["A"] == max(by_letter.values())

    def test_summary_dataframe_shape(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """summary_dataframe() returns 5 rows with the expected columns."""
        report = engine.run(ten_hcp_df)
        df = report.summary_dataframe()
        assert len(df) == 5
        for col in (
            "letter", "hcp_count", "total_rx",
            "mean_rx", "min_rx", "max_rx", "rx_share_pct",
        ):
            assert col in df.columns


# ---------------------------------------------------------------------------
# 5. Full pipeline / run()
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests for the end-to-end run() method."""

    def test_run_returns_decile_report(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """run() returns a DecileReport instance."""
        report = engine.run(ten_hcp_df)
        assert isinstance(report, DecileReport)

    def test_run_total_hcps_matches_input(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """report.total_hcps equals input row count."""
        report = engine.run(twenty_hcp_df)
        assert report.total_hcps == len(twenty_hcp_df)

    def test_run_total_rx_matches_input(
        self, engine: DecileTieringEngine, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """report.total_rx equals the sum of the Rx column."""
        expected = float(twenty_hcp_df["prescriptions_last_12m"].sum())
        report = engine.run(twenty_hcp_df)
        assert abs(report.total_rx - expected) < 0.01

    def test_run_data_contains_both_columns(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """report.data contains both rx_decile and letter_tier."""
        report = engine.run(ten_hcp_df)
        assert "rx_decile" in report.data.columns
        assert "letter_tier" in report.data.columns

    def test_run_does_not_mutate_input(
        self, engine: DecileTieringEngine, ten_hcp_df: pd.DataFrame
    ) -> None:
        """run() does not modify the caller's DataFrame."""
        before = ten_hcp_df.copy()
        engine.run(ten_hcp_df)
        pd.testing.assert_frame_equal(before, ten_hcp_df)


# ---------------------------------------------------------------------------
# 6. Functional API
# ---------------------------------------------------------------------------

class TestFunctionalAPI:
    """Tests for the module-level convenience functions."""

    def test_assign_decile_tiers_adds_both_columns(
        self, ten_hcp_df: pd.DataFrame
    ) -> None:
        """assign_decile_tiers() returns both decile and letter columns."""
        result = assign_decile_tiers(ten_hcp_df)
        assert "rx_decile" in result.columns
        assert "letter_tier" in result.columns

    def test_letter_distribution_returns_dataframe(
        self, twenty_hcp_df: pd.DataFrame
    ) -> None:
        """letter_distribution() returns a 5-row DataFrame."""
        df = letter_distribution(twenty_hcp_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_custom_rx_column_name(self) -> None:
        """Custom Rx column name flows through the functional API."""
        df = pd.DataFrame(
            {
                "npi": [f"N{i}" for i in range(10)],
                "nrx": [500, 450, 400, 350, 300, 250, 200, 150, 100, 50],
            }
        )
        result = assign_decile_tiers(df, rx_column="nrx", id_column="npi")
        assert "letter_tier" in result.columns
        assert result["letter_tier"].tolist() == [
            "A", "A", "B", "B", "C", "C", "D", "D", "E", "E",
        ]


# ---------------------------------------------------------------------------
# 7. Dataclass behaviour
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for LetterSummary and DecileReport immutability."""

    def test_letter_summary_is_immutable(self) -> None:
        """LetterSummary is a frozen dataclass — attribute assignment fails."""
        s = LetterSummary(
            letter="A",
            hcp_count=2,
            total_rx=950.0,
            mean_rx=475.0,
            min_rx=450.0,
            max_rx=500.0,
            rx_share_pct=35.0,
        )
        with pytest.raises((AttributeError, Exception)):
            s.letter = "B"  # type: ignore[misc]

    def test_default_rx_column_exposed(self) -> None:
        """The module exposes a DEFAULT_RX_COLUMN constant."""
        assert DEFAULT_RX_COLUMN == "prescriptions_last_12m"

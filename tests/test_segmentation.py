"""
Unit tests for HCPSegmentationEngine.

Run with::

    pytest tests/ -v

or with coverage::

    pytest tests/ -v --cov=src --cov-report=term-missing

Coverage targets (minimum 80%):
- Segmentation logic (tiering / segment assignment)
- Composite score calculation
- Input validation and type checking
- Edge cases: empty dataset, single HCP, all-identical scores, missing columns
- Filtering helpers (specialty, region)
- Analysis and summary helpers
- Pipeline stages and the full pipeline
- Export / to_dataframe utility
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root when running without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import (
    HCPSegmentationEngine,
    TIER_THRESHOLDS,
    SCORE_WEIGHTS,
    REQUIRED_COLUMNS,
    _normalise_columns,
    _min_max_normalise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine() -> HCPSegmentationEngine:
    """Return a default HCPSegmentationEngine instance."""
    return HCPSegmentationEngine()


@pytest.fixture()
def full_sample_df() -> pd.DataFrame:
    """Return a realistic 20-row HCP DataFrame covering all tiers/segments."""
    return pd.DataFrame(
        {
            "hcp_id": [f"HCP{i:03d}" for i in range(1, 21)],
            "name": [f"Dr. Test {i}" for i in range(1, 21)],
            "specialty": (
                ["Cardiology"] * 5
                + ["Oncology"] * 5
                + ["Neurology"] * 5
                + ["Primary Care"] * 5
            ),
            "city": ["New York"] * 10 + ["Los Angeles"] * 10,
            "region": ["Northeast"] * 10 + ["West"] * 10,
            "prescriptions_last_12m": [
                350, 320, 280, 210, 180,
                150, 130, 110, 90, 80,
                60, 55, 45, 40, 35,
                20, 15, 10, 5, 2,
            ],
            "total_rx_value_usd": [
                130000, 120000, 105000, 78750, 67500,
                56250, 48750, 41250, 33750, 30000,
                22500, 20625, 16875, 15000, 13125,
                7500, 5625, 3750, 1875, 750,
            ],
            "num_visits": [8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            "digital_engagement_score": [
                88, 80, 75, 65, 60,
                72, 55, 50, 65, 40,
                45, 38, 30, 25, 20,
                18, 15, 12, 8, 5,
            ],
            "kol_flag": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tier": [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
            "segment": [
                "High-Value KOL", "High-Value KOL", "Growth Target", "Growth Target",
                "Digital Adopter", "Digital Adopter", "Standard", "Standard",
                "Standard", "Standard", "Low Activity", "Low Activity",
                "Low Activity", "Low Activity", "Dormant", "Dormant",
                "Dormant", "Dormant", "Dormant", "Dormant",
            ],
        }
    )


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Return the smallest valid DataFrame (one HCP record)."""
    return pd.DataFrame(
        {
            "hcp_id": ["HCP001"],
            "specialty": ["Cardiology"],
            "region": ["Northeast"],
            "prescriptions_last_12m": [200],
            "total_rx_value_usd": [75000.0],
            "num_visits": [5],
            "digital_engagement_score": [70],
            "kol_flag": [1],
        }
    )


@pytest.fixture()
def scored_df(engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame) -> pd.DataFrame:
    """Return full_sample_df with composite_score already computed."""
    return engine.calculate_scores(full_sample_df)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for module-level helper utilities."""

    def test_normalise_columns_lowercases(self) -> None:
        """_normalise_columns must lowercase all names."""
        result = _normalise_columns(["HCP_ID", "Specialty", "REGION"])
        assert result == ["hcp_id", "specialty", "region"]

    def test_normalise_columns_replaces_spaces(self) -> None:
        """_normalise_columns must replace spaces with underscores."""
        result = _normalise_columns(["Total Rx Value", "Num Visits"])
        assert result == ["total_rx_value", "num_visits"]

    def test_normalise_columns_does_not_mutate_input(self) -> None:
        """_normalise_columns must not modify the input list."""
        original = ["HCP ID", "Specialty"]
        _normalise_columns(original)
        assert original == ["HCP ID", "Specialty"]

    def test_min_max_normalise_standard(self) -> None:
        """_min_max_normalise must return values in [0, 1]."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _min_max_normalise(series)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_min_max_normalise_zero_range(self) -> None:
        """_min_max_normalise must return all zeros for constant series."""
        series = pd.Series([5.0, 5.0, 5.0])
        result = _min_max_normalise(series)
        assert (result == 0.0).all()

    def test_min_max_normalise_single_element(self) -> None:
        """_min_max_normalise on a single-element series returns 0.0."""
        series = pd.Series([42.0])
        result = _min_max_normalise(series)
        assert result.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

class TestEngineInit:
    """Tests for HCPSegmentationEngine.__init__()."""

    def test_default_config_uses_module_constants(self) -> None:
        """Default engine must use module-level constants."""
        eng = HCPSegmentationEngine()
        assert eng._tier_thresholds == TIER_THRESHOLDS
        assert eng._score_weights == SCORE_WEIGHTS
        assert eng._required_columns == REQUIRED_COLUMNS

    def test_custom_config_overrides_defaults(self) -> None:
        """Custom config values must override defaults."""
        custom_tiers = {1: 500, 2: 200, 3: 80, 4: 0}
        eng = HCPSegmentationEngine(config={"tier_thresholds": custom_tiers})
        assert eng._tier_thresholds == custom_tiers

    def test_none_config_uses_defaults(self) -> None:
        """Passing config=None must produce the same result as no config."""
        eng = HCPSegmentationEngine(config=None)
        assert eng._tier_thresholds == TIER_THRESHOLDS

    def test_invalid_config_type_raises(self) -> None:
        """Passing a non-dict config must raise TypeError."""
        with pytest.raises(TypeError, match="config must be a dict"):
            HCPSegmentationEngine(config="bad_config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for HCPSegmentationEngine.validate()."""

    def test_validate_raises_on_empty_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """validate() should raise ValueError for an empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            engine.validate(pd.DataFrame())

    def test_validate_raises_on_missing_required_columns(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """validate() should raise ValueError when required columns are absent."""
        df = pd.DataFrame({"irrelevant_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Required columns missing"):
            engine.validate(df)

    def test_validate_passes_with_required_columns(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """validate() should return True for a well-formed DataFrame."""
        assert engine.validate(full_sample_df) is True

    def test_validate_raises_on_none(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """validate() should raise TypeError when passed None."""
        with pytest.raises(TypeError):
            engine.validate(None)  # type: ignore[arg-type]

    def test_validate_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """validate() should raise TypeError for non-DataFrame input."""
        with pytest.raises(TypeError):
            engine.validate([{"hcp_id": "HCP001"}])  # type: ignore[arg-type]

    def test_validate_column_names_are_case_insensitive(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """validate() must accept columns regardless of capitalisation."""
        df = pd.DataFrame(
            {
                "HCP_ID": ["H1"],
                "SPECIALTY": ["Cardiology"],
                "PRESCRIPTIONS_LAST_12M": [100],
                "DIGITAL_ENGAGEMENT_SCORE": [50],
            }
        )
        assert engine.validate(df) is True

    def test_load_data_raises_on_missing_file(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """load_data() should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            engine.load_data("/nonexistent/path/data.csv")

    def test_load_data_raises_on_unsupported_extension(
        self, engine: HCPSegmentationEngine, tmp_path: Path
    ) -> None:
        """load_data() should raise ValueError for unsupported file formats."""
        bad_file = tmp_path / "data.parquet"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            engine.load_data(str(bad_file))

    def test_load_data_raises_on_empty_filepath(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """load_data() should raise ValueError for an empty filepath string."""
        with pytest.raises(ValueError):
            engine.load_data("")

    def test_load_data_csv_roundtrip(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_data() must load a CSV file and return the correct shape."""
        csv_path = tmp_path / "test_hcp.csv"
        full_sample_df.to_csv(str(csv_path), index=False)
        loaded = engine.load_data(str(csv_path))
        assert loaded.shape == full_sample_df.shape

    def test_load_data_excel_roundtrip(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_data() must load an Excel file and return the correct shape."""
        xlsx_path = tmp_path / "test_hcp.xlsx"
        full_sample_df.to_excel(str(xlsx_path), index=False)
        loaded = engine.load_data(str(xlsx_path))
        assert loaded.shape == full_sample_df.shape


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for HCPSegmentationEngine.preprocess()."""

    def test_preprocess_drops_fully_null_rows(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must remove entirely-null rows."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", None],
                "specialty": ["Cardiology", None],
                "prescriptions_last_12m": [100, None],
                "digital_engagement_score": [50, None],
            }
        )
        result = engine.preprocess(df)
        assert len(result) == 1

    def test_preprocess_does_not_mutate_input(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """preprocess() must not modify the original DataFrame."""
        original_len = len(full_sample_df)
        original_cols = list(full_sample_df.columns)
        engine.preprocess(full_sample_df)
        assert len(full_sample_df) == original_len
        assert list(full_sample_df.columns) == original_cols

    def test_preprocess_normalises_column_names(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must standardise column names to lower_snake_case."""
        df = pd.DataFrame(
            {
                "HCP ID": ["A"],
                "Specialty": ["Cardiology"],
                "Prescriptions Last 12M": [100],
                "Digital Engagement Score": [50],
            }
        )
        result = engine.preprocess(df)
        assert "hcp_id" in result.columns
        assert "specialty" in result.columns

    def test_preprocess_fills_numeric_nulls_with_median(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must fill NaN numeric values with column medians."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", "B", "C"],
                "specialty": ["Cardiology"] * 3,
                "prescriptions_last_12m": [100.0, None, 200.0],
                "digital_engagement_score": [50, 60, 70],
            }
        )
        result = engine.preprocess(df)
        assert result["prescriptions_last_12m"].notna().all()
        assert result["prescriptions_last_12m"].iloc[1] == pytest.approx(150.0)

    def test_preprocess_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must raise TypeError for non-DataFrame inputs."""
        with pytest.raises(TypeError):
            engine.preprocess({"data": [1, 2, 3]})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Score calculation
# ---------------------------------------------------------------------------

class TestScoreCalculation:
    """Tests for HCPSegmentationEngine.calculate_scores()."""

    def test_composite_score_column_created(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """calculate_scores() must add a 'composite_score' column."""
        result = engine.calculate_scores(full_sample_df)
        assert "composite_score" in result.columns

    def test_composite_score_range(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """All composite scores must fall within [0, 100]."""
        result = engine.calculate_scores(full_sample_df)
        assert result["composite_score"].between(0, 100).all(), (
            f"Scores out of range:\n{result['composite_score'].describe()}"
        )

    def test_higher_rx_yields_higher_score(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """The HCP with the most prescriptions should have the highest score."""
        result = engine.calculate_scores(full_sample_df)
        top_rx_idx = full_sample_df["prescriptions_last_12m"].idxmax()
        top_score_idx = result["composite_score"].idxmax()
        assert top_rx_idx == top_score_idx

    def test_original_dataframe_not_mutated(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """calculate_scores() must not mutate the input DataFrame."""
        original_cols = list(full_sample_df.columns)
        engine.calculate_scores(full_sample_df)
        assert list(full_sample_df.columns) == original_cols

    def test_all_identical_scores_do_not_raise(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """calculate_scores() should handle all-identical values without error."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", "B", "C"],
                "specialty": ["Cardiology"] * 3,
                "prescriptions_last_12m": [100, 100, 100],
                "total_rx_value_usd": [50000.0, 50000.0, 50000.0],
                "num_visits": [5, 5, 5],
                "digital_engagement_score": [60, 60, 60],
            }
        )
        result = engine.calculate_scores(df)
        assert (result["composite_score"] == 0.0).all()

    def test_single_hcp_score_calculation(
        self, engine: HCPSegmentationEngine, minimal_df: pd.DataFrame
    ) -> None:
        """calculate_scores() must succeed for a single-row DataFrame."""
        result = engine.calculate_scores(minimal_df)
        assert "composite_score" in result.columns
        assert len(result) == 1

    def test_missing_score_columns_return_zero(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """calculate_scores() must return 0.0 when no score columns are present."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", "B"],
                "specialty": ["Cardiology", "Oncology"],
                "region": ["Northeast", "West"],
            }
        )
        result = engine.calculate_scores(df)
        assert "composite_score" in result.columns
        assert (result["composite_score"] == 0.0).all()

    def test_score_weights_partial_columns(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """calculate_scores() must redistribute weights for missing columns."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", "B", "C"],
                "specialty": ["Cardiology"] * 3,
                "prescriptions_last_12m": [100, 200, 300],
                "digital_engagement_score": [50, 60, 70],
            }
        )
        result = engine.calculate_scores(df)
        # Scores should still be in [0, 100]
        assert result["composite_score"].between(0, 100).all()

    def test_calculate_scores_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """calculate_scores() must raise TypeError for non-DataFrame inputs."""
        with pytest.raises(TypeError):
            engine.calculate_scores("not a dataframe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment:
    """Tests for HCPSegmentationEngine.assign_tiers()."""

    def test_tier_1_assigned_above_threshold(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with >= 300 prescriptions must receive Tier 1."""
        df = pd.DataFrame(
            {
                "hcp_id": ["T1"],
                "specialty": ["Cardiology"],
                "prescriptions_last_12m": [350],
                "digital_engagement_score": [80],
            }
        )
        result = engine.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 1

    def test_tier_4_assigned_below_threshold(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with < 50 prescriptions must receive Tier 4."""
        df = pd.DataFrame(
            {
                "hcp_id": ["T4"],
                "specialty": ["Primary Care"],
                "prescriptions_last_12m": [10],
                "digital_engagement_score": [20],
            }
        )
        result = engine.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 4

    def test_all_tier_boundaries(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """Verify tier boundaries for all four tiers."""
        rx_values = [400, 150, 60, 20]
        expected_tiers = [1, 2, 3, 4]
        df = pd.DataFrame(
            {
                "hcp_id": [f"H{i}" for i in range(4)],
                "specialty": ["Cardiology"] * 4,
                "prescriptions_last_12m": rx_values,
                "digital_engagement_score": [50] * 4,
            }
        )
        result = engine.assign_tiers(df)
        assert list(result["computed_tier"]) == expected_tiers

    def test_tier_boundary_exact_values(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """Verify exact boundary values fall in the correct tier."""
        # Exactly at each boundary
        df = pd.DataFrame(
            {
                "hcp_id": ["B1", "B2", "B3", "B4"],
                "specialty": ["Cardiology"] * 4,
                "prescriptions_last_12m": [300, 100, 50, 49],
                "digital_engagement_score": [50] * 4,
            }
        )
        result = engine.assign_tiers(df)
        tiers = list(result["computed_tier"])
        assert tiers[0] == 1  # exactly 300 -> Tier 1
        assert tiers[1] == 2  # exactly 100 -> Tier 2
        assert tiers[2] == 3  # exactly 50  -> Tier 3
        assert tiers[3] == 4  # 49          -> Tier 4

    def test_assign_tiers_immutability(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """assign_tiers() must not mutate the input DataFrame."""
        original_cols = list(full_sample_df.columns)
        engine.assign_tiers(full_sample_df)
        assert list(full_sample_df.columns) == original_cols

    def test_assign_tiers_raises_on_missing_rx_column(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """assign_tiers() must raise KeyError when prescriptions_last_12m is absent."""
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "specialty": ["Cardiology"],
                "digital_engagement_score": [50],
            }
        )
        with pytest.raises(KeyError, match="prescriptions_last_12m"):
            engine.assign_tiers(df)

    def test_assign_tiers_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """assign_tiers() must raise TypeError for non-DataFrame inputs."""
        with pytest.raises(TypeError):
            engine.assign_tiers(None)  # type: ignore[arg-type]

    def test_custom_tier_thresholds_via_config(self) -> None:
        """Custom tier_thresholds in config must override defaults."""
        custom_config = {
            "tier_thresholds": {1: 500, 2: 200, 3: 80, 4: 0},
            "required_columns": [
                "hcp_id",
                "specialty",
                "prescriptions_last_12m",
                "digital_engagement_score",
            ],
        }
        eng = HCPSegmentationEngine(config=custom_config)
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "specialty": ["Oncology"],
                "prescriptions_last_12m": [350],  # < 500 -> Tier 2 with custom config
                "digital_engagement_score": [60],
            }
        )
        result = eng.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 2


# ---------------------------------------------------------------------------
# 5. Segmentation logic
# ---------------------------------------------------------------------------

class TestSegmentation:
    """Tests for HCPSegmentationEngine.segment()."""

    def test_segment_raises_without_composite_score(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """segment() should raise KeyError if composite_score is absent."""
        with pytest.raises(KeyError, match="composite_score"):
            engine.segment(full_sample_df)

    def test_kol_high_value_segment(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """KOL HCPs with composite_score >= 70 receive 'High-Value KOL'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["K1"],
                "specialty": ["Oncology"],
                "prescriptions_last_12m": [400],
                "digital_engagement_score": [85],
                "kol_flag": [1],
                "composite_score": [90.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "High-Value KOL"

    def test_non_kol_cannot_be_high_value_kol(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """Non-KOL HCPs should never receive 'High-Value KOL' segment."""
        df = pd.DataFrame(
            {
                "hcp_id": ["NK1"],
                "specialty": ["Cardiology"],
                "prescriptions_last_12m": [400],
                "digital_engagement_score": [85],
                "kol_flag": [0],
                "composite_score": [95.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] != "High-Value KOL"

    def test_growth_target_segment(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with composite_score >= 60 and kol_flag=0 -> 'Growth Target'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["G1"],
                "specialty": ["Cardiology"],
                "digital_engagement_score": [30],
                "kol_flag": [0],
                "composite_score": [65.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "Growth Target"

    def test_digital_adopter_segment(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with digital_engagement_score >= 60 but low composite -> 'Digital Adopter'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["DA1"],
                "specialty": ["Neurology"],
                "digital_engagement_score": [75],
                "kol_flag": [0],
                "composite_score": [35.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "Digital Adopter"

    def test_standard_segment(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with composite_score >= 30 and digital < 60 -> 'Standard'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["S1"],
                "specialty": ["Primary Care"],
                "digital_engagement_score": [40],
                "kol_flag": [0],
                "composite_score": [45.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "Standard"

    def test_low_activity_segment(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with 10 <= composite_score < 30 and digital < 60 -> 'Low Activity'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["LA1"],
                "specialty": ["Primary Care"],
                "digital_engagement_score": [20],
                "kol_flag": [0],
                "composite_score": [15.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "Low Activity"

    def test_dormant_segment_for_lowest_scores(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """HCPs with near-zero scores should be classified as 'Dormant'."""
        df = pd.DataFrame(
            {
                "hcp_id": ["D1"],
                "specialty": ["Primary Care"],
                "prescriptions_last_12m": [2],
                "digital_engagement_score": [3],
                "kol_flag": [0],
                "composite_score": [2.0],
            }
        )
        result = engine.segment(df)
        assert result["computed_segment"].iloc[0] == "Dormant"

    def test_segment_without_kol_flag_column(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """segment() must succeed when kol_flag column is absent (defaults to 0)."""
        df = pd.DataFrame(
            {
                "hcp_id": ["NK2"],
                "specialty": ["Cardiology"],
                "digital_engagement_score": [30],
                "composite_score": [80.0],
            }
        )
        result = engine.segment(df)
        # Without kol_flag, even score=80 cannot produce High-Value KOL
        assert result["computed_segment"].iloc[0] != "High-Value KOL"

    def test_segment_immutability(
        self, engine: HCPSegmentationEngine, scored_df: pd.DataFrame
    ) -> None:
        """segment() must not mutate the input DataFrame."""
        original_cols = list(scored_df.columns)
        engine.segment(scored_df)
        assert list(scored_df.columns) == original_cols

    def test_segment_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """segment() must raise TypeError for non-DataFrame inputs."""
        with pytest.raises(TypeError):
            engine.segment(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Tests for HCPSegmentationEngine.run_full_pipeline()."""

    def test_full_pipeline_produces_all_columns(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must add composite_score, computed_tier, computed_segment."""
        result = engine.run_full_pipeline(full_sample_df)
        for col in ("composite_score", "computed_tier", "computed_segment"):
            assert col in result.columns, f"Missing column: {col}"

    def test_full_pipeline_single_hcp(
        self, engine: HCPSegmentationEngine, minimal_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must succeed for a single-row DataFrame."""
        result = engine.run_full_pipeline(minimal_df)
        assert len(result) == 1
        assert "computed_segment" in result.columns

    def test_full_pipeline_does_not_mutate_input(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must not modify the caller's DataFrame."""
        original_cols = list(full_sample_df.columns)
        original_len = len(full_sample_df)
        engine.run_full_pipeline(full_sample_df)
        assert list(full_sample_df.columns) == original_cols
        assert len(full_sample_df) == original_len

    def test_full_pipeline_raises_on_empty_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """run_full_pipeline() must raise ValueError for an empty DataFrame."""
        with pytest.raises(ValueError):
            engine.run_full_pipeline(pd.DataFrame())

    def test_full_pipeline_raises_on_missing_required_columns(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """run_full_pipeline() must raise ValueError for DataFrames missing required columns."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})
        with pytest.raises(ValueError, match="Required columns missing"):
            engine.run_full_pipeline(df)

    def test_full_pipeline_segment_values_are_valid(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must assign only known segment names."""
        valid_segments = {
            "High-Value KOL",
            "Growth Target",
            "Digital Adopter",
            "Standard",
            "Low Activity",
            "Dormant",
        }
        result = engine.run_full_pipeline(full_sample_df)
        assigned = set(result["computed_segment"].unique())
        assert assigned.issubset(valid_segments), f"Unknown segments: {assigned - valid_segments}"

    def test_full_pipeline_tier_values_are_1_to_4(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must assign only tier values 1-4."""
        result = engine.run_full_pipeline(full_sample_df)
        assert result["computed_tier"].isin([1, 2, 3, 4]).all()

    def test_full_pipeline_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """run_full_pipeline() must raise TypeError for non-DataFrame inputs."""
        with pytest.raises(TypeError):
            engine.run_full_pipeline(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. Filtering helpers
# ---------------------------------------------------------------------------

class TestFiltering:
    """Tests for filter_by_specialty() and filter_by_region()."""

    def test_filter_by_specialty_returns_correct_subset(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must return only rows matching the specialty."""
        result = engine.filter_by_specialty(full_sample_df, "Cardiology")
        assert (result["specialty"] == "Cardiology").all()
        assert len(result) == 5

    def test_filter_by_specialty_case_insensitive(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must be case-insensitive."""
        lower = engine.filter_by_specialty(full_sample_df, "cardiology")
        upper = engine.filter_by_specialty(full_sample_df, "CARDIOLOGY")
        assert len(lower) == len(upper) == 5

    def test_filter_by_specialty_returns_empty_for_unknown(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() should return an empty DataFrame for unknown specialty."""
        result = engine.filter_by_specialty(full_sample_df, "Dentistry")
        assert result.empty

    def test_filter_by_region_returns_correct_subset(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_region() must return only rows matching the region."""
        result = engine.filter_by_region(full_sample_df, "Northeast")
        assert (result["region"] == "Northeast").all()
        assert len(result) == 10

    def test_filter_by_region_case_insensitive(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_region() must be case-insensitive."""
        lower = engine.filter_by_region(full_sample_df, "northeast")
        upper = engine.filter_by_region(full_sample_df, "NORTHEAST")
        assert len(lower) == len(upper) == 10

    def test_filter_by_specialty_does_not_mutate_input(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must not modify the original DataFrame."""
        original_len = len(full_sample_df)
        engine.filter_by_specialty(full_sample_df, "Oncology")
        assert len(full_sample_df) == original_len

    def test_filter_by_region_does_not_mutate_input(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_region() must not modify the original DataFrame."""
        original_len = len(full_sample_df)
        engine.filter_by_region(full_sample_df, "West")
        assert len(full_sample_df) == original_len

    def test_filter_by_specialty_raises_on_missing_column(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """filter_by_specialty() must raise KeyError when specialty column is absent."""
        df = pd.DataFrame({"hcp_id": ["H1"], "region": ["Northeast"]})
        with pytest.raises(KeyError, match="specialty"):
            engine.filter_by_specialty(df, "Cardiology")

    def test_filter_by_region_raises_on_missing_column(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """filter_by_region() must raise KeyError when region column is absent."""
        df = pd.DataFrame({"hcp_id": ["H1"], "specialty": ["Cardiology"]})
        with pytest.raises(KeyError, match="region"):
            engine.filter_by_region(df, "Northeast")

    def test_filter_by_specialty_raises_on_empty_string(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must raise ValueError for an empty specialty string."""
        with pytest.raises(ValueError):
            engine.filter_by_specialty(full_sample_df, "")

    def test_filter_by_region_raises_on_empty_string(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_region() must raise ValueError for an empty region string."""
        with pytest.raises(ValueError):
            engine.filter_by_region(full_sample_df, "")


# ---------------------------------------------------------------------------
# 8. Analysis helpers
# ---------------------------------------------------------------------------

class TestAnalysis:
    """Tests for analyze() and get_segment_summary()."""

    def test_analyze_returns_required_keys(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """analyze() result must contain total_records, columns, missing_pct."""
        result = engine.analyze(full_sample_df)
        for key in ("total_records", "columns", "missing_pct"):
            assert key in result, f"Missing key: {key}"

    def test_analyze_total_records_matches_input(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """analyze() total_records must match the number of rows."""
        result = engine.analyze(full_sample_df)
        assert result["total_records"] == len(full_sample_df)

    def test_analyze_raises_on_non_dataframe(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """analyze() must raise TypeError for non-DataFrame input."""
        with pytest.raises(TypeError):
            engine.analyze("not a dataframe")  # type: ignore[arg-type]

    def test_get_segment_summary_raises_without_computed_segment(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """get_segment_summary() must raise KeyError when computed_segment is absent."""
        with pytest.raises(KeyError, match="computed_segment"):
            engine.get_segment_summary(full_sample_df)

    def test_get_segment_summary_returns_expected_columns(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """get_segment_summary() must include 'count' and 'avg_composite_score' columns."""
        result_df = engine.run_full_pipeline(full_sample_df)
        summary = engine.get_segment_summary(result_df)
        assert "count" in summary.columns
        assert "avg_composite_score" in summary.columns

    def test_get_segment_summary_count_sums_to_total(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """get_segment_summary() counts must sum to total number of HCPs."""
        result_df = engine.run_full_pipeline(full_sample_df)
        summary = engine.get_segment_summary(result_df)
        assert summary["count"].sum() == len(result_df)

    def test_to_dataframe_flattens_nested_dict(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """to_dataframe() must expand nested dicts with dot-notation keys."""
        result = {"totals": {"rx": 500, "visits": 10}, "count": 20}
        df = engine.to_dataframe(result)
        metrics = list(df["metric"])
        assert "totals.rx" in metrics
        assert "totals.visits" in metrics
        assert "count" in metrics

    def test_to_dataframe_returns_metric_and_value_columns(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """to_dataframe() must return exactly 'metric' and 'value' columns."""
        result_dict = {"total_records": 10}
        df = engine.to_dataframe(result_dict)
        assert list(df.columns) == ["metric", "value"]

    def test_to_dataframe_raises_on_non_dict(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """to_dataframe() must raise TypeError for non-dict input."""
        with pytest.raises(TypeError):
            engine.to_dataframe([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge-case and robustness tests."""

    def test_large_prescription_counts_tier_1(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """Very large prescription counts must still produce Tier 1."""
        df = pd.DataFrame(
            {
                "hcp_id": ["MEGA"],
                "specialty": ["Oncology"],
                "prescriptions_last_12m": [999999],
                "digital_engagement_score": [90],
            }
        )
        result = engine.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 1

    def test_zero_prescriptions_tier_4(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """Zero prescription count must produce Tier 4."""
        df = pd.DataFrame(
            {
                "hcp_id": ["ZERO"],
                "specialty": ["Primary Care"],
                "prescriptions_last_12m": [0],
                "digital_engagement_score": [20],
            }
        )
        result = engine.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 4

    def test_full_pipeline_preserves_original_row_count(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """run_full_pipeline() must not drop any rows from a clean dataset."""
        result = engine.run_full_pipeline(full_sample_df)
        assert len(result) == len(full_sample_df)

    def test_preprocess_preserves_non_null_rows(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must keep rows that have at least one non-null value."""
        df = pd.DataFrame(
            {
                "hcp_id": ["A", "B", "C"],
                "specialty": ["Cardiology", None, "Neurology"],
                "prescriptions_last_12m": [100, None, 200],
                "digital_engagement_score": [50, None, 70],
            }
        )
        result = engine.preprocess(df)
        # Row "B" has non-null hcp_id? Actually hcp_id is None too — row should survive
        # because not ALL fields are null.
        # Row B: hcp_id=None, specialty=None, rx=None, digital=None -> all null -> dropped
        assert "A" in result["hcp_id"].values
        assert "C" in result["hcp_id"].values

    def test_run_method_returns_analyze_dict(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """run() convenience wrapper must return a dict with total_records key."""
        csv_path = tmp_path / "hcp_run_test.csv"
        full_sample_df.to_csv(str(csv_path), index=False)
        result = engine.run(str(csv_path))
        assert "total_records" in result

    def test_multiple_pipelines_are_independent(
        self, full_sample_df: pd.DataFrame
    ) -> None:
        """Two independent engine instances must not share state."""
        eng1 = HCPSegmentationEngine()
        eng2 = HCPSegmentationEngine(config={"tier_thresholds": {1: 500, 2: 200, 3: 80, 4: 0}})
        r1 = eng1.run_full_pipeline(full_sample_df)
        r2 = eng2.run_full_pipeline(full_sample_df)
        # Tier assignments may differ due to different thresholds
        assert not (r1["computed_tier"] == r2["computed_tier"]).all()

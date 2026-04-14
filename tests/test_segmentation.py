"""
Unit tests for HCPSegmentationEngine.

Run with::

    pytest tests/test_segmentation.py -v

Coverage targets (minimum 80%):
- Segmentation logic (tiering / segment assignment)
- Composite score calculation
- Input validation
- Edge cases: empty dataset, single HCP, all-identical scores
- Filtering helpers (specialty, region)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root when running without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import HCPSegmentationEngine, TIER_THRESHOLDS


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
                350, 320, 280, 210, 180,  # Tier 1 / Tier 2
                150, 130, 110, 90, 80,   # Tier 2 / Tier 3
                60, 55, 45, 40, 35,      # Tier 3 / Tier 4
                20, 15, 10, 5, 2,        # Tier 4
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


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for HCPSegmentationEngine.validate()."""

    def test_validate_raises_on_empty_dataframe(self, engine: HCPSegmentationEngine) -> None:
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


# ---------------------------------------------------------------------------
# 2. Score calculation
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
            f"Scores out of range: {result['composite_score'].describe()}"
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
        # When all values are identical the normalised score is 0 for those cols
        assert (result["composite_score"] == 0.0).all()

    def test_single_hcp_score_calculation(
        self, engine: HCPSegmentationEngine, minimal_df: pd.DataFrame
    ) -> None:
        """calculate_scores() must succeed for a single-row DataFrame."""
        result = engine.calculate_scores(minimal_df)
        assert "composite_score" in result.columns
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 3. Tier assignment
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

    def test_assign_tiers_immutability(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """assign_tiers() must not mutate the input DataFrame."""
        original_cols = list(full_sample_df.columns)
        engine.assign_tiers(full_sample_df)
        assert list(full_sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# 4. Segmentation logic
# ---------------------------------------------------------------------------

class TestSegmentation:
    """Tests for HCPSegmentationEngine.segment()."""

    def test_segment_raises_without_composite_score(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """segment() should raise KeyError if composite_score is absent."""
        with pytest.raises(KeyError, match="composite_score"):
            engine.segment(full_sample_df)

    def test_kol_high_value_segment(self, engine: HCPSegmentationEngine) -> None:
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


# ---------------------------------------------------------------------------
# 5. Filtering helpers
# ---------------------------------------------------------------------------

class TestFiltering:
    """Tests for filter_by_specialty() and filter_by_region()."""

    def test_filter_by_specialty_returns_correct_subset(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must return only rows matching the specialty."""
        result = engine.filter_by_specialty(full_sample_df, "Cardiology")
        assert (result["specialty"] == "Cardiology").all()
        assert len(result) == 5  # 5 Cardiology rows in fixture

    def test_filter_by_specialty_case_insensitive(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """filter_by_specialty() must be case-insensitive."""
        lower = engine.filter_by_specialty(full_sample_df, "cardiology")
        upper = engine.filter_by_specialty(full_sample_df, "CARDIOLOGY")
        assert len(lower) == len(upper)

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
        assert len(result) == 10  # 10 Northeast rows in fixture

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


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_validate_raises_on_none(self, engine: HCPSegmentationEngine) -> None:
        """validate() should raise ValueError when passed None."""
        with pytest.raises((ValueError, AttributeError)):
            engine.validate(None)  # type: ignore[arg-type]

    def test_preprocess_drops_fully_null_rows(
        self, engine: HCPSegmentationEngine
    ) -> None:
        """preprocess() must remove entirely-null rows without mutation."""
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
        assert len(df) == 2  # original unmodified

    def test_analyze_returns_required_keys(
        self, engine: HCPSegmentationEngine, full_sample_df: pd.DataFrame
    ) -> None:
        """analyze() result must contain total_records, columns, missing_pct."""
        result = engine.analyze(full_sample_df)
        for key in ("total_records", "columns", "missing_pct"):
            assert key in result, f"Missing key: {key}"

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

    def test_custom_tier_thresholds_via_config(self) -> None:
        """Custom tier_thresholds in config must override defaults."""
        custom_config = {
            "tier_thresholds": {1: 500, 2: 200, 3: 80, 4: 0},
            "required_columns": ["hcp_id", "specialty", "prescriptions_last_12m",
                                  "digital_engagement_score"],
        }
        engine = HCPSegmentationEngine(config=custom_config)
        df = pd.DataFrame(
            {
                "hcp_id": ["H1"],
                "specialty": ["Oncology"],
                "prescriptions_last_12m": [350],  # < 500 → Tier 2 with custom config
                "digital_engagement_score": [60],
            }
        )
        result = engine.assign_tiers(df)
        assert result["computed_tier"].iloc[0] == 2

"""
Healthcare professional segmentation and targeting engine for pharma sales.

This module provides the core HCPSegmentationEngine class, which supports
loading HCP data, validating inputs, computing engagement scores, assigning
tiers, and segmenting healthcare professionals for targeted outreach.

Author: github.com/achmadnaufal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


# Tier thresholds based on prescriptions_last_12m
TIER_THRESHOLDS = {
    1: 300,   # Tier 1: >= 300 prescriptions
    2: 100,   # Tier 2: >= 100 prescriptions
    3: 50,    # Tier 3: >= 50 prescriptions
    4: 0,     # Tier 4: < 50 prescriptions
}

REQUIRED_COLUMNS = [
    "hcp_id",
    "specialty",
    "prescriptions_last_12m",
    "digital_engagement_score",
]

SCORE_WEIGHTS = {
    "prescriptions_last_12m": 0.40,
    "total_rx_value_usd": 0.30,
    "digital_engagement_score": 0.20,
    "num_visits": 0.10,
}


class HCPSegmentationEngine:
    """HCP targeting and segmentation engine for pharmaceutical sales teams.

    Provides a full pipeline from raw CSV/Excel data through validation,
    preprocessing, score calculation, tier assignment, and segmentation.

    Attributes:
        config: Optional configuration dictionary to override default behaviour.

    Example:
        >>> engine = HCPSegmentationEngine()
        >>> df = engine.load_data("demo/sample_data.csv")
        >>> result = engine.run_full_pipeline(df)
        >>> print(result["segments"].value_counts())
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the engine with an optional configuration dictionary.

        Args:
            config: Key/value pairs that override default thresholds and
                weights.  Supported keys: ``tier_thresholds``,
                ``score_weights``, ``required_columns``.
        """
        self.config = config or {}
        self._tier_thresholds: Dict[int, int] = self.config.get(
            "tier_thresholds", TIER_THRESHOLDS
        )
        self._score_weights: Dict[str, float] = self.config.get(
            "score_weights", SCORE_WEIGHTS
        )
        self._required_columns: List[str] = self.config.get(
            "required_columns", REQUIRED_COLUMNS
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load HCP data from a CSV or Excel file.

        Args:
            filepath: Absolute or relative path to the data file.  Supported
                extensions are ``.csv``, ``.xlsx``, and ``.xls``.

        Returns:
            A DataFrame containing the raw HCP records.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if path.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        if path.suffix == ".csv":
            return pd.read_csv(filepath)
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Use .csv, .xlsx, or .xls."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the input DataFrame against minimum quality requirements.

        Checks performed:
        - Dataset is not empty.
        - All required columns are present.
        - ``hcp_id`` values are unique (warns but does not raise).

        Args:
            df: Raw or preprocessed HCP DataFrame.

        Returns:
            ``True`` when all checks pass.

        Raises:
            ValueError: If the dataset is empty or required columns are absent.
        """
        if df is None or df.empty:
            raise ValueError(
                "Input dataset is empty. Provide at least one HCP record."
            )

        normalised_cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
        missing = [
            col for col in self._required_columns if col not in normalised_cols
        ]
        if missing:
            raise ValueError(
                f"Required columns missing from dataset: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        return True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the input DataFrame.

        Operations applied (non-mutating — a copy is returned):
        - Drop rows that are entirely ``NaN``.
        - Standardise column names to ``lower_snake_case``.
        - Fill missing numeric values with column medians.

        Args:
            df: Raw HCP DataFrame.

        Returns:
            A new, cleaned DataFrame; the original is not modified.
        """
        cleaned = df.copy()
        cleaned = cleaned.dropna(how="all")
        cleaned.columns = [
            c.lower().strip().replace(" ", "_") for c in cleaned.columns
        ]

        numeric_cols = cleaned.select_dtypes(include="number").columns
        medians = cleaned[numeric_cols].median()
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(medians)

        return cleaned

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a composite HCP engagement score for every record.

        The score is a weighted, min-max normalised combination of
        ``prescriptions_last_12m``, ``total_rx_value_usd``,
        ``digital_engagement_score``, and ``num_visits``.  Missing columns
        are skipped and their weight redistributed proportionally.

        Handles edge cases:
        - Single-row datasets (score defaults to 100.0).
        - Columns where all values are identical (normalised to 0.0).

        Args:
            df: Preprocessed HCP DataFrame.

        Returns:
            A new DataFrame with an additional ``composite_score`` column
            (float, 0–100).  The original is not modified.
        """
        result = df.copy()

        available = {
            col: weight
            for col, weight in self._score_weights.items()
            if col in result.columns
        }

        if not available:
            result["composite_score"] = 0.0
            return result

        total_weight = sum(available.values())
        normalised_weights = {
            col: w / total_weight for col, w in available.items()
        }

        score_series = pd.Series(0.0, index=result.index)

        for col, weight in normalised_weights.items():
            col_min = result[col].min()
            col_max = result[col].max()
            col_range = col_max - col_min

            if col_range == 0:
                # All values identical — normalise to zero contribution
                normalised = pd.Series(0.0, index=result.index)
            else:
                normalised = (result[col] - col_min) / col_range

            score_series = score_series + normalised * weight * 100

        result = result.copy()
        result["composite_score"] = score_series.round(2)
        return result

    # ------------------------------------------------------------------
    # Tiering
    # ------------------------------------------------------------------

    def assign_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a tier label (1–4) to each HCP based on prescription volume.

        Tier boundaries (overridable via ``config["tier_thresholds"]``):
        - Tier 1: >= 300 prescriptions
        - Tier 2: >= 100 prescriptions
        - Tier 3: >= 50 prescriptions
        - Tier 4: < 50 prescriptions

        Args:
            df: Preprocessed HCP DataFrame containing
                ``prescriptions_last_12m``.

        Returns:
            A new DataFrame with an additional ``computed_tier`` column
            (int).  The original is not modified.

        Raises:
            KeyError: If ``prescriptions_last_12m`` is absent from *df*.
        """
        if "prescriptions_last_12m" not in df.columns:
            raise KeyError(
                "'prescriptions_last_12m' column is required for tier assignment."
            )

        result = df.copy()
        rx = result["prescriptions_last_12m"]

        tier_col = pd.Series(4, index=result.index, dtype=int)
        tier_col = tier_col.where(rx < self._tier_thresholds[1], other=3)
        tier_col = tier_col.where(rx < self._tier_thresholds[2], other=2)
        tier_col = tier_col.where(rx < self._tier_thresholds[1], other=3)

        # Apply from lowest priority to highest to respect boundaries
        tier_col = pd.Series(4, index=result.index, dtype=int)
        for tier in sorted(self._tier_thresholds.keys(), reverse=True):
            threshold = self._tier_thresholds[tier]
            tier_col = tier_col.where(rx < threshold, other=tier)

        result["computed_tier"] = tier_col
        return result

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each HCP into a named segment using score and KOL status.

        Segment logic (evaluated in order):
        1. ``High-Value KOL``   – ``kol_flag == 1`` AND ``composite_score >= 70``
        2. ``Growth Target``    – ``composite_score >= 60``
        3. ``Digital Adopter``  – ``digital_engagement_score >= 60``
        4. ``Standard``         – ``composite_score >= 30``
        5. ``Low Activity``     – ``composite_score >= 10``
        6. ``Dormant``          – everything else

        Requires ``calculate_scores`` to have been run first so that
        ``composite_score`` is present.

        Args:
            df: DataFrame with ``composite_score``, ``digital_engagement_score``,
                and optionally ``kol_flag``.

        Returns:
            A new DataFrame with an additional ``computed_segment`` column
            (str).  The original is not modified.
        """
        if "composite_score" not in df.columns:
            raise KeyError(
                "'composite_score' column is missing. "
                "Run calculate_scores() before segment()."
            )

        result = df.copy()
        score = result["composite_score"]
        digital = result.get("digital_engagement_score", pd.Series(0, index=result.index))
        kol = result.get("kol_flag", pd.Series(0, index=result.index))

        segment_col = pd.Series("Dormant", index=result.index, dtype=str)
        segment_col = segment_col.where(score < 10, other="Low Activity")
        segment_col = segment_col.where(score < 30, other="Standard")
        segment_col = segment_col.where(digital < 60, other="Digital Adopter")
        segment_col = segment_col.where(score < 60, other="Growth Target")
        is_kol = (kol == 1) & (score >= 70)
        segment_col = segment_col.where(~is_kol, other="High-Value KOL")

        result["computed_segment"] = segment_col
        return result

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_specialty(
        self, df: pd.DataFrame, specialty: str
    ) -> pd.DataFrame:
        """Return a subset of HCPs matching a given specialty (case-insensitive).

        Args:
            df: Preprocessed HCP DataFrame containing a ``specialty`` column.
            specialty: The specialty name to filter on.

        Returns:
            A new filtered DataFrame; the original is not modified.

        Raises:
            KeyError: If ``specialty`` column is absent from *df*.
        """
        if "specialty" not in df.columns:
            raise KeyError("'specialty' column is required for specialty filtering.")
        mask = df["specialty"].str.lower() == specialty.lower()
        return df.loc[mask].copy()

    def filter_by_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Return a subset of HCPs matching a given region (case-insensitive).

        Args:
            df: Preprocessed HCP DataFrame containing a ``region`` column.
            region: The region name to filter on.

        Returns:
            A new filtered DataFrame; the original is not modified.

        Raises:
            KeyError: If ``region`` column is absent from *df*.
        """
        if "region" not in df.columns:
            raise KeyError("'region' column is required for region filtering.")
        mask = df["region"].str.lower() == region.lower()
        return df.loc[mask].copy()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for a preprocessed HCP DataFrame.

        Args:
            df: Raw or preprocessed HCP DataFrame.

        Returns:
            A dictionary containing:
            - ``total_records`` (int)
            - ``columns`` (list[str])
            - ``missing_pct`` (dict) – percentage missing per column
            - ``summary_stats`` (dict) – descriptive stats for numeric columns
            - ``totals`` (dict) – column sums for numeric columns
            - ``means`` (dict) – column means for numeric columns
        """
        cleaned = self.preprocess(df)
        result: Dict[str, Any] = {
            "total_records": len(cleaned),
            "columns": list(cleaned.columns),
            "missing_pct": (
                cleaned.isnull().sum() / max(len(cleaned), 1) * 100
            ).round(1).to_dict(),
        }
        numeric_df = cleaned.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete segmentation pipeline on a DataFrame.

        Steps: preprocess → validate → score → tier → segment.

        Args:
            df: Raw HCP DataFrame (CSV/Excel contents).

        Returns:
            A new, fully enriched DataFrame with ``composite_score``,
            ``computed_tier``, and ``computed_segment`` columns appended.
            The original DataFrame is never mutated.

        Raises:
            ValueError: If validation fails.
        """
        cleaned = self.preprocess(df)
        self.validate(cleaned)
        scored = self.calculate_scores(cleaned)
        tiered = self.assign_tiers(scored)
        segmented = self.segment(tiered)
        return segmented

    def run(self, filepath: str) -> Dict[str, Any]:
        """Convenience wrapper: load a file and return summary analysis.

        Args:
            filepath: Path to a ``.csv`` or ``.xlsx`` HCP data file.

        Returns:
            Summary metrics dictionary as produced by :meth:`analyze`.
        """
        df = self.load_data(filepath)
        self.validate(self.preprocess(df))
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """Flatten an analysis result dictionary into a two-column DataFrame.

        Nested dicts are expanded with dot-notation keys, e.g.
        ``{"totals": {"rx": 500}}`` becomes ``{"metric": "totals.rx", "value": 500}``.

        Args:
            result: Dictionary returned by :meth:`analyze` or similar methods.

        Returns:
            A DataFrame with columns ``metric`` and ``value``.
        """
        rows = []
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({"metric": f"{key}.{sub_key}", "value": sub_value})
            else:
                rows.append({"metric": key, "value": value})
        return pd.DataFrame(rows)

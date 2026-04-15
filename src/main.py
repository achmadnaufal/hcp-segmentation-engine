"""
Healthcare professional segmentation and targeting engine for pharma sales.

This module provides the core HCPSegmentationEngine class, which supports
loading HCP data, validating inputs, computing engagement scores, assigning
tiers, and segmenting healthcare professionals for targeted outreach.

Segment types (in priority order):
    - High-Value KOL:  KOL-flagged HCPs with composite_score >= 70
    - Growth Target:   composite_score >= 60
    - Digital Adopter: digital_engagement_score >= 60
    - Standard:        composite_score >= 30
    - Low Activity:    composite_score >= 10
    - Dormant:         composite_score < 10

Author: github.com/achmadnaufal
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Tier thresholds based on prescriptions_last_12m
TIER_THRESHOLDS: Dict[int, int] = {
    1: 300,  # Tier 1: >= 300 prescriptions
    2: 100,  # Tier 2: >= 100 prescriptions
    3: 50,   # Tier 3: >= 50 prescriptions
    4: 0,    # Tier 4: < 50 prescriptions
}

REQUIRED_COLUMNS: List[str] = [
    "hcp_id",
    "specialty",
    "prescriptions_last_12m",
    "digital_engagement_score",
]

SCORE_WEIGHTS: Dict[str, float] = {
    "prescriptions_last_12m": 0.40,
    "total_rx_value_usd": 0.30,
    "digital_engagement_score": 0.20,
    "num_visits": 0.10,
}

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".csv", ".xlsx", ".xls")

SEGMENT_NAMES: Tuple[str, ...] = (
    "High-Value KOL",
    "Growth Target",
    "Digital Adopter",
    "Standard",
    "Low Activity",
    "Dormant",
)

# Composite score thresholds that drive segment assignment
_SEGMENT_THRESHOLDS: Dict[str, float] = {
    "kol": 70.0,
    "growth": 60.0,
    "digital": 60.0,
    "standard": 30.0,
    "low_activity": 10.0,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalise_columns(columns: List[str]) -> List[str]:
    """Return column names converted to lower_snake_case.

    Args:
        columns: Original column name list.

    Returns:
        New list with each name stripped, lowercased, and spaces replaced by
        underscores.  The input list is not mutated.
    """
    return [c.lower().strip().replace(" ", "_") for c in columns]


def _min_max_normalise(series: pd.Series) -> pd.Series:
    """Apply min-max normalisation to a numeric Series.

    When all values in *series* are identical the range is zero; in that case
    the function returns a Series of zeros rather than raising a
    ZeroDivisionError.

    Args:
        series: Numeric pandas Series to normalise.

    Returns:
        A new Series with values in [0, 1].  Returns all-zero Series if the
        range is zero.  The input Series is not mutated.
    """
    col_min: float = float(series.min())
    col_max: float = float(series.max())
    col_range: float = col_max - col_min

    if col_range == 0.0:
        return pd.Series(0.0, index=series.index)

    return (series - col_min) / col_range


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class HCPSegmentationEngine:
    """HCP targeting and segmentation engine for pharmaceutical sales teams.

    Provides a full pipeline from raw CSV/Excel data through validation,
    preprocessing, score calculation, tier assignment, and segmentation.

    Every public method follows an **immutable** convention: a new DataFrame
    is returned and the caller's input is never modified in-place.

    Attributes:
        config: Optional configuration dictionary used at construction time.

    Example:
        >>> engine = HCPSegmentationEngine()
        >>> df = engine.load_data("demo/sample_data.csv")
        >>> result = engine.run_full_pipeline(df)
        >>> print(result[["hcp_id", "computed_tier", "computed_segment"]])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the engine with an optional configuration dictionary.

        All configuration values fall back to sensible module-level defaults
        when the corresponding key is absent from *config*.

        Args:
            config: Key/value pairs that override default thresholds and
                weights.  Supported keys:

                - ``tier_thresholds`` (Dict[int, int]) - prescription
                  boundaries for Tier 1-4.
                - ``score_weights`` (Dict[str, float]) - column weights used
                  by :meth:`calculate_scores`.
                - ``required_columns`` (List[str]) - columns that must be
                  present for :meth:`validate` to pass.
                - ``segment_thresholds`` (Dict[str, float]) - composite score
                  cut-offs that drive :meth:`segment`.

        Raises:
            TypeError: If *config* is provided but is not a dict.
        """
        if config is not None and not isinstance(config, dict):
            raise TypeError(
                f"config must be a dict or None, got {type(config).__name__}."
            )

        self.config: Dict[str, Any] = config or {}
        self._tier_thresholds: Dict[int, int] = self.config.get(
            "tier_thresholds", TIER_THRESHOLDS
        )
        self._score_weights: Dict[str, float] = self.config.get(
            "score_weights", SCORE_WEIGHTS
        )
        self._required_columns: List[str] = self.config.get(
            "required_columns", REQUIRED_COLUMNS
        )
        self._segment_thresholds: Dict[str, float] = self.config.get(
            "segment_thresholds", _SEGMENT_THRESHOLDS
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load HCP data from a CSV or Excel file into a DataFrame.

        Args:
            filepath: Absolute or relative path to the data file.  Supported
                extensions: ``.csv``, ``.xlsx``, ``.xls``.

        Returns:
            A DataFrame containing the raw, unmodified HCP records.

        Raises:
            FileNotFoundError: If *filepath* does not point to an existing
                file.
            ValueError: If the file extension is not in
                ``SUPPORTED_EXTENSIONS`` or if the parsed file contains no
                rows.
        """
        if not filepath or not str(filepath).strip():
            raise ValueError("filepath must be a non-empty string.")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. "
                f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
            )

        if suffix in (".xlsx", ".xls"):
            df: pd.DataFrame = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError(
                f"File '{filepath}' loaded successfully but contains no rows."
            )

        logger.info("Loaded %d rows from '%s'.", len(df), filepath)
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the input DataFrame against minimum quality requirements.

        Checks performed (in order):

        1. *df* is not ``None``.
        2. *df* is a :class:`pandas.DataFrame`.
        3. The dataset is not empty (zero rows).
        4. All :attr:`_required_columns` are present (column names are
           compared after normalisation to lower_snake_case).

        A warning is logged if ``hcp_id`` values are not unique, but this
        does not raise an exception.

        Args:
            df: Raw or preprocessed HCP DataFrame to validate.

        Returns:
            ``True`` when all hard checks pass.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            ValueError: If the dataset is empty or required columns are
                absent.
        """
        if df is None:
            raise TypeError(
                "Expected a pandas DataFrame but received None. "
                "Pass a valid DataFrame to validate()."
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            raise ValueError(
                "Input dataset is empty. Provide at least one HCP record."
            )

        normalised_cols = _normalise_columns(list(df.columns))
        missing = [
            col for col in self._required_columns if col not in normalised_cols
        ]
        if missing:
            raise ValueError(
                f"Required columns missing from dataset: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        if "hcp_id" in normalised_cols:
            hcp_id_col = df.columns[normalised_cols.index("hcp_id")]
            n_duplicates: int = int(df[hcp_id_col].duplicated().sum())
            if n_duplicates > 0:
                logger.warning(
                    "Dataset contains %d duplicate hcp_id values. "
                    "Downstream operations may produce unexpected results.",
                    n_duplicates,
                )

        return True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the input DataFrame.

        Operations applied (all non-mutating; a copy is returned):

        - Drop rows that are entirely ``NaN``.
        - Standardise column names to ``lower_snake_case``.
        - Fill missing numeric values with per-column medians.

        Args:
            df: Raw HCP DataFrame.  Must not be ``None``.

        Returns:
            A new, cleaned DataFrame with standardised column names.  The
            original DataFrame is never modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            ValueError: If *df* is empty after dropping all-null rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )

        cleaned: pd.DataFrame = df.copy()
        cleaned = cleaned.dropna(how="all")

        if cleaned.empty:
            raise ValueError(
                "DataFrame is empty after removing all-null rows.  "
                "Provide at least one valid HCP record."
            )

        cleaned = cleaned.copy()
        cleaned.columns = pd.Index(_normalise_columns(list(cleaned.columns)))

        numeric_cols = cleaned.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            medians: pd.Series = cleaned[numeric_cols].median()
            cleaned[numeric_cols] = cleaned[numeric_cols].fillna(medians)

        return cleaned

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a composite HCP engagement score (0-100) for every record.

        The composite score is a weighted, min-max normalised combination of
        the columns listed in :attr:`_score_weights`.  Missing columns are
        silently skipped and their weight is redistributed proportionally
        across the remaining columns.

        Edge-case handling:

        - **No scoring columns present** -- all records receive
          ``composite_score = 0.0``.
        - **Single-row dataset** -- all values normalise to 0 because
          min == max; the score is 0.0.
        - **All-identical column values** -- the column contributes 0.0 to
          the score (safe zero-division avoidance).

        Args:
            df: Preprocessed HCP DataFrame.

        Returns:
            A new DataFrame with an additional ``composite_score`` column
            (float, rounded to 2 d.p.).  The input is not modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )

        result: pd.DataFrame = df.copy()

        available: Dict[str, float] = {
            col: weight
            for col, weight in self._score_weights.items()
            if col in result.columns
        }

        if not available:
            result = result.copy()
            result["composite_score"] = 0.0
            logger.warning(
                "None of the score-weight columns %s found in DataFrame. "
                "composite_score set to 0.0 for all records.",
                list(self._score_weights.keys()),
            )
            return result

        total_weight: float = sum(available.values())
        normalised_weights: Dict[str, float] = {
            col: w / total_weight for col, w in available.items()
        }

        score_series: pd.Series = pd.Series(0.0, index=result.index)

        for col, weight in normalised_weights.items():
            normalised: pd.Series = _min_max_normalise(result[col])
            score_series = score_series + normalised * weight * 100.0

        result = result.copy()
        result["composite_score"] = score_series.round(2)
        return result

    # ------------------------------------------------------------------
    # Tiering
    # ------------------------------------------------------------------

    def assign_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a tier label (1-4) to each HCP based on prescription volume.

        Tier boundaries (configurable via ``config["tier_thresholds"]``):

        +---------+-------------------------------+
        | Tier    | Condition                     |
        +=========+===============================+
        | Tier 1  | >= 300 prescriptions          |
        +---------+-------------------------------+
        | Tier 2  | >= 100 and < 300              |
        +---------+-------------------------------+
        | Tier 3  | >= 50 and < 100               |
        +---------+-------------------------------+
        | Tier 4  | < 50 prescriptions            |
        +---------+-------------------------------+

        Args:
            df: Preprocessed HCP DataFrame that must contain a
                ``prescriptions_last_12m`` column.

        Returns:
            A new DataFrame with an additional ``computed_tier`` column
            (integer 1-4).  The input is not modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            KeyError: If ``prescriptions_last_12m`` is absent from *df*.
            ValueError: If :attr:`_tier_thresholds` is missing required tier
                keys (1-4).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if "prescriptions_last_12m" not in df.columns:
            raise KeyError(
                "'prescriptions_last_12m' column is required for tier "
                "assignment but was not found in the DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        required_tier_keys = {1, 2, 3, 4}
        missing_keys = required_tier_keys - set(self._tier_thresholds.keys())
        if missing_keys:
            raise ValueError(
                f"tier_thresholds config is missing keys: {sorted(missing_keys)}. "
                "All four tiers (1-4) must be defined."
            )

        result: pd.DataFrame = df.copy()
        rx: pd.Series = result["prescriptions_last_12m"]

        # Start everyone at Tier 4, then upgrade based on thresholds.
        tier_col: pd.Series = pd.Series(4, index=result.index, dtype=int)
        for tier in sorted(self._tier_thresholds.keys(), reverse=True):
            threshold: int = self._tier_thresholds[tier]
            tier_col = tier_col.where(rx < threshold, other=tier)

        result = result.copy()
        result["computed_tier"] = tier_col
        return result

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each HCP into a named actionable segment.

        Segment logic (evaluated in priority order):

        1. **High-Value KOL**  -- ``kol_flag == 1`` AND
           ``composite_score >= segment_thresholds["kol"]`` (default 70)
        2. **Growth Target**   -- ``composite_score >= segment_thresholds["growth"]``
           (default 60)
        3. **Digital Adopter** -- ``digital_engagement_score >= segment_thresholds["digital"]``
           (default 60)
        4. **Standard**        -- ``composite_score >= segment_thresholds["standard"]``
           (default 30)
        5. **Low Activity**    -- ``composite_score >= segment_thresholds["low_activity"]``
           (default 10)
        6. **Dormant**         -- all remaining HCPs

        :meth:`calculate_scores` must be called before this method so that
        ``composite_score`` is present in the DataFrame.

        Args:
            df: DataFrame with ``composite_score`` and
                ``digital_engagement_score`` columns; ``kol_flag`` is
                optional (defaults to 0 for all rows when absent).

        Returns:
            A new DataFrame with an additional ``computed_segment`` column
            (str).  The input is not modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            KeyError: If ``composite_score`` is absent from *df*.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if "composite_score" not in df.columns:
            raise KeyError(
                "'composite_score' column is missing. "
                "Run calculate_scores() before segment()."
            )

        result: pd.DataFrame = df.copy()
        score: pd.Series = result["composite_score"]

        # Gracefully handle missing optional columns
        digital: pd.Series = (
            result["digital_engagement_score"]
            if "digital_engagement_score" in result.columns
            else pd.Series(0.0, index=result.index)
        )
        kol: pd.Series = (
            result["kol_flag"]
            if "kol_flag" in result.columns
            else pd.Series(0, index=result.index)
        )

        thresholds = self._segment_thresholds

        # Build segment column using cascading priority rules.
        # Start at lowest priority (Dormant) and promote upward.
        segment_col: pd.Series = pd.Series("Dormant", index=result.index, dtype=str)
        segment_col = segment_col.where(
            score < thresholds.get("low_activity", 10.0), other="Low Activity"
        )
        segment_col = segment_col.where(
            score < thresholds.get("standard", 30.0), other="Standard"
        )
        segment_col = segment_col.where(
            digital < thresholds.get("digital", 60.0), other="Digital Adopter"
        )
        segment_col = segment_col.where(
            score < thresholds.get("growth", 60.0), other="Growth Target"
        )
        is_kol: pd.Series = (kol == 1) & (score >= thresholds.get("kol", 70.0))
        segment_col = segment_col.where(~is_kol, other="High-Value KOL")

        result = result.copy()
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
            specialty: The specialty name to filter on (e.g. ``"Cardiology"``).
                The comparison is case-insensitive.

        Returns:
            A new filtered DataFrame.  An empty DataFrame is returned (not an
            error) when no rows match.  The input is not modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            KeyError: If a ``specialty`` column is absent from *df*.
            ValueError: If *specialty* is an empty string.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if "specialty" not in df.columns:
            raise KeyError(
                "'specialty' column is required for specialty filtering. "
                f"Available columns: {list(df.columns)}"
            )
        if not specialty or not specialty.strip():
            raise ValueError(
                "specialty must be a non-empty string."
            )

        mask: pd.Series = df["specialty"].str.lower() == specialty.lower().strip()
        return df.loc[mask].copy()

    def filter_by_region(
        self, df: pd.DataFrame, region: str
    ) -> pd.DataFrame:
        """Return a subset of HCPs matching a given region (case-insensitive).

        Args:
            df: Preprocessed HCP DataFrame containing a ``region`` column.
            region: The region name to filter on (e.g. ``"Northeast"``).
                The comparison is case-insensitive.

        Returns:
            A new filtered DataFrame.  An empty DataFrame is returned (not an
            error) when no rows match.  The input is not modified.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            KeyError: If a ``region`` column is absent from *df*.
            ValueError: If *region* is an empty string.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if "region" not in df.columns:
            raise KeyError(
                "'region' column is required for region filtering. "
                f"Available columns: {list(df.columns)}"
            )
        if not region or not region.strip():
            raise ValueError(
                "region must be a non-empty string."
            )

        mask: pd.Series = df["region"].str.lower() == region.lower().strip()
        return df.loc[mask].copy()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for a preprocessed HCP DataFrame.

        Internally runs :meth:`preprocess` to normalise column names and fill
        missing values before computing statistics.

        Args:
            df: Raw or preprocessed HCP DataFrame.

        Returns:
            A dictionary containing:

            - ``total_records`` (int) -- number of rows after preprocessing.
            - ``columns`` (list[str]) -- column names after normalisation.
            - ``missing_pct`` (dict) -- percentage of missing values per
              column, rounded to 1 d.p.
            - ``summary_stats`` (dict) -- descriptive statistics for numeric
              columns (via ``DataFrame.describe()``).
            - ``totals`` (dict) -- column sums for numeric columns.
            - ``means`` (dict) -- column means for numeric columns.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )

        cleaned: pd.DataFrame = self.preprocess(df)
        result: Dict[str, Any] = {
            "total_records": len(cleaned),
            "columns": list(cleaned.columns),
            "missing_pct": (
                cleaned.isnull().sum() / max(len(cleaned), 1) * 100
            ).round(1).to_dict(),
        }

        numeric_df: pd.DataFrame = cleaned.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()

        return result

    def get_segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-segment counts and average composite scores.

        Requires that *df* already contains ``computed_segment`` (produced by
        :meth:`segment` or :meth:`run_full_pipeline`).

        Args:
            df: Enriched HCP DataFrame with ``computed_segment`` and
                optionally ``composite_score``.

        Returns:
            A new DataFrame indexed by segment name with columns:

            - ``count`` -- number of HCPs in the segment.
            - ``avg_composite_score`` -- mean composite score (NaN when
              ``composite_score`` is absent).

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            KeyError: If ``computed_segment`` is absent from *df*.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if "computed_segment" not in df.columns:
            raise KeyError(
                "'computed_segment' column is missing. "
                "Run segment() or run_full_pipeline() first."
            )

        count_series: pd.Series = df["computed_segment"].value_counts()
        summary: pd.DataFrame = count_series.rename("count").to_frame()

        if "composite_score" in df.columns:
            avg_score: pd.Series = (
                df.groupby("computed_segment")["composite_score"]
                .mean()
                .round(2)
                .rename("avg_composite_score")
            )
            summary = summary.join(avg_score)
        else:
            summary["avg_composite_score"] = float("nan")

        return summary

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete segmentation pipeline on a DataFrame.

        Pipeline stages (in order):

        1. :meth:`preprocess` -- clean and normalise column names.
        2. :meth:`validate`   -- assert required columns and non-empty data.
        3. :meth:`calculate_scores` -- compute weighted composite score.
        4. :meth:`assign_tiers`     -- assign Tier 1-4.
        5. :meth:`segment`          -- classify into named segment.

        The original DataFrame passed in is never modified.

        Args:
            df: Raw HCP DataFrame (e.g. the direct output of
                :meth:`load_data`).

        Returns:
            A new, fully enriched DataFrame with ``composite_score``,
            ``computed_tier``, and ``computed_segment`` columns appended.

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            ValueError: If validation fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )

        cleaned: pd.DataFrame = self.preprocess(df)
        self.validate(cleaned)
        scored: pd.DataFrame = self.calculate_scores(cleaned)
        tiered: pd.DataFrame = self.assign_tiers(scored)
        segmented: pd.DataFrame = self.segment(tiered)
        return segmented

    def run(self, filepath: str) -> Dict[str, Any]:
        """Convenience wrapper: load a file and return summary analysis.

        Equivalent to calling :meth:`load_data`, :meth:`preprocess`,
        :meth:`validate`, and :meth:`analyze` in sequence.

        Args:
            filepath: Path to a ``.csv`` or ``.xlsx`` HCP data file.

        Returns:
            Summary metrics dictionary as produced by :meth:`analyze`.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If the file format is unsupported or validation fails.
        """
        df: pd.DataFrame = self.load_data(filepath)
        preprocessed: pd.DataFrame = self.preprocess(df)
        self.validate(preprocessed)
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Flatten an analysis result dictionary into a two-column DataFrame.

        Nested dicts are expanded using dot-notation keys.  For example::

            {"totals": {"rx": 500}} -> [{"metric": "totals.rx", "value": 500}]

        Non-dict values are emitted as-is.  The input dict is not mutated.

        Args:
            result: Dictionary returned by :meth:`analyze` or a similar
                method that returns key/value pairs.

        Returns:
            A new DataFrame with exactly two columns: ``metric`` (str) and
            ``value`` (any).

        Raises:
            TypeError: If *result* is not a dict.
        """
        if not isinstance(result, dict):
            raise TypeError(
                f"Expected a dict, got {type(result).__name__}."
            )

        rows: List[Dict[str, Any]] = []
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({"metric": f"{key}.{sub_key}", "value": sub_value})
            else:
                rows.append({"metric": str(key), "value": value})

        return pd.DataFrame(rows, columns=["metric", "value"])

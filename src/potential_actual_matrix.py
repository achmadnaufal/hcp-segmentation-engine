"""
Potential-vs-Actual matrix (a.k.a. bubble segmentation) for HCPs.

This module classifies Healthcare Professionals on a 2x2 strategic grid
comparing their **potential** prescribing capacity against their **actual**
current prescribing volume. The output quadrants directly drive commercial
investment decisions:

- **Star**       — High potential, High actual.  Protect and expand share
                   of voice; these HCPs are both valuable and engaged.
- **Grow**       — High potential, Low actual.  Biggest conversion
                   opportunity; allocate incremental promotional effort.
- **Maintain**   — Low potential, High actual.  Over-indexed on actuals
                   relative to ceiling; protect at lower cost.
- **Monitor**    — Low potential, Low actual.  Minimum investment; watch
                   for behavioural change.

The split between high and low is driven by quantile cut-offs (default =
median on each axis). A per-HCP ``gap_score`` (potential minus actual,
both min-max normalised to [0, 100]) surfaces the biggest untapped-demand
HCPs at the top of the sales prioritisation list.

Every function is **immutable** — input DataFrames are never mutated; a
new DataFrame is always returned.

Author: github.com/achmadnaufal
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical quadrant labels in priority order (highest commercial value first).
QUADRANT_LABELS: Tuple[str, ...] = ("Star", "Grow", "Maintain", "Monitor")

#: Default split quantile (median) used to dichotomise potential and actual.
DEFAULT_SPLIT_QUANTILE: float = 0.5

#: Default column names expected on input DataFrames.
DEFAULT_POTENTIAL_COL: str = "potential_score"
DEFAULT_ACTUAL_COL: str = "rx_volume_last_12m"

#: Fallback actual column if the preferred column is absent (keeps the module
#: usable against the legacy ``prescriptions_last_12m`` schema).
FALLBACK_ACTUAL_COLS: Tuple[str, ...] = (
    "rx_volume_last_12m",
    "prescriptions_last_12m",
    "rx_volume_last_6m",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_actual_column(df: pd.DataFrame, preferred: str) -> str:
    """Return the first column from ``FALLBACK_ACTUAL_COLS`` present in *df*.

    If *preferred* is in *df* it wins.  Otherwise fall back in order so that
    the module can operate against either the legacy
    ``prescriptions_last_12m`` schema or the extended
    ``rx_volume_last_12m`` schema without forcing the caller to rename
    columns.

    Args:
        df: DataFrame whose columns should be searched.
        preferred: Caller-supplied column name (tried first).

    Returns:
        The resolved column name.

    Raises:
        KeyError: If none of the candidate columns are present.
    """
    if preferred in df.columns:
        return preferred
    for candidate in FALLBACK_ACTUAL_COLS:
        if candidate in df.columns:
            logger.debug(
                "Preferred actual column '%s' absent; falling back to '%s'.",
                preferred,
                candidate,
            )
            return candidate
    raise KeyError(
        f"No actual-volume column found.  Tried '{preferred}' and fallbacks "
        f"{list(FALLBACK_ACTUAL_COLS)}.  Available columns: {list(df.columns)}"
    )


def _min_max_0_100(series: pd.Series) -> pd.Series:
    """Rescale a numeric Series to the ``[0, 100]`` range.

    Constant or single-row Series degenerate to all-zero output rather than
    raising a ZeroDivisionError; this keeps the downstream ``gap_score``
    well-defined in edge cases.

    Args:
        series: Numeric values to rescale.

    Returns:
        A new Series aligned with ``series.index`` in the inclusive range
        ``[0, 100]``.
    """
    numeric: pd.Series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    col_min: float = float(numeric.min())
    col_max: float = float(numeric.max())
    col_range: float = col_max - col_min

    if col_range == 0.0:
        return pd.Series(0.0, index=series.index)

    return ((numeric - col_min) / col_range * 100.0).astype(float)


def _classify_quadrant(
    potential_hi: bool, actual_hi: bool
) -> str:
    """Return the quadrant label for a (potential, actual) boolean pair.

    Args:
        potential_hi: True when the HCP sits above the potential split.
        actual_hi: True when the HCP sits above the actual split.

    Returns:
        One of ``"Star"``, ``"Grow"``, ``"Maintain"``, ``"Monitor"``.
    """
    if potential_hi and actual_hi:
        return "Star"
    if potential_hi and not actual_hi:
        return "Grow"
    if not potential_hi and actual_hi:
        return "Maintain"
    return "Monitor"


def _split_threshold(
    series: pd.Series, quantile: float
) -> float:
    """Compute the split threshold for a numeric Series using a quantile.

    Returns the series mean when every value is identical (ensuring every
    HCP is classified as "low" in that degenerate case rather than crashing).

    Args:
        series: Numeric values to split.
        quantile: Value in ``[0, 1]``.

    Returns:
        The numeric threshold.
    """
    numeric: pd.Series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.nunique() <= 1:
        return float(numeric.mean()) if len(numeric) else 0.0
    return float(numeric.quantile(quantile))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_potential_actual_matrix(
    df: pd.DataFrame,
    potential_col: str = DEFAULT_POTENTIAL_COL,
    actual_col: str = DEFAULT_ACTUAL_COL,
    split_quantile: float = DEFAULT_SPLIT_QUANTILE,
) -> pd.DataFrame:
    """Classify each HCP into a potential-vs-actual quadrant.

    The function returns a NEW DataFrame with these columns appended:

    - ``potential_norm`` — potential_col min-max normalised to [0, 100].
    - ``actual_norm``    — actual_col min-max normalised to [0, 100].
    - ``gap_score``      — ``potential_norm - actual_norm`` (can be negative
                           when the HCP over-performs their ceiling).
    - ``quadrant``       — one of ``"Star"``, ``"Grow"``, ``"Maintain"``,
                           ``"Monitor"``.
    - ``potential_tier`` — ``"High"`` or ``"Low"`` after quantile split.
    - ``actual_tier``    — ``"High"`` or ``"Low"`` after quantile split.

    Edge cases handled without raising:

    - Empty DataFrame (zero rows): returns an empty DataFrame with all new
      columns present.
    - Single HCP: quadrant defaults to ``"Monitor"`` because there is no
      cohort to split against.
    - All-zero or constant potential / actual: every HCP is classified as
      ``"Monitor"`` and ``gap_score`` is 0.0.
    - NaN values in either column: coerced to 0 before splitting.

    Args:
        df: HCP-level DataFrame.  Must contain a potential column and at
            least one fallback actual-volume column (see
            :data:`FALLBACK_ACTUAL_COLS`).
        potential_col: Name of the potential-score column.  Defaults to
            ``"potential_score"``.
        actual_col: Name of the actual-volume column.  Defaults to
            ``"rx_volume_last_12m"``; automatically falls back to
            ``"prescriptions_last_12m"`` when absent.
        split_quantile: Quantile in ``[0, 1]`` used to dichotomise each
            axis.  Defaults to 0.5 (median split).

    Returns:
        A new DataFrame containing every input column plus the six
        enrichment columns listed above.  The input is not modified.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.
        KeyError: If neither *potential_col* nor any fallback actual
            column is present in *df*.
        ValueError: If *split_quantile* is not in ``(0, 1)``.

    Example:
        >>> import pandas as pd
        >>> from src.potential_actual_matrix import compute_potential_actual_matrix
        >>> df = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3", "H4"],
        ...     "potential_score": [90, 80, 30, 20],
        ...     "rx_volume_last_12m": [300, 50, 250, 30],
        ... })
        >>> result = compute_potential_actual_matrix(df)
        >>> sorted(result["quadrant"].unique())
        ['Grow', 'Maintain', 'Monitor', 'Star']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )
    if not (0.0 < split_quantile < 1.0):
        raise ValueError(
            f"split_quantile must be in (0, 1), got {split_quantile}."
        )

    empty_columns: List[str] = [
        "potential_norm",
        "actual_norm",
        "gap_score",
        "quadrant",
        "potential_tier",
        "actual_tier",
    ]

    # Empty input: return a schema-preserving empty DataFrame.
    if df.empty:
        empty: pd.DataFrame = df.copy()
        for col in empty_columns:
            empty[col] = pd.Series(dtype="object" if col in {"quadrant", "potential_tier", "actual_tier"} else "float64")
        logger.info("compute_potential_actual_matrix received an empty DataFrame.")
        return empty

    if potential_col not in df.columns:
        raise KeyError(
            f"Potential column '{potential_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    resolved_actual: str = _resolve_actual_column(df, actual_col)

    work: pd.DataFrame = df.copy()

    potential_norm: pd.Series = _min_max_0_100(work[potential_col])
    actual_norm: pd.Series = _min_max_0_100(work[resolved_actual])
    gap_score: pd.Series = (potential_norm - actual_norm).round(2)

    potential_threshold: float = _split_threshold(
        work[potential_col], split_quantile
    )
    actual_threshold: float = _split_threshold(
        work[resolved_actual], split_quantile
    )

    potential_values: pd.Series = pd.to_numeric(
        work[potential_col], errors="coerce"
    ).fillna(0.0)
    actual_values: pd.Series = pd.to_numeric(
        work[resolved_actual], errors="coerce"
    ).fillna(0.0)

    # Strict "greater than" so a degenerate all-zero column keeps every row
    # in the low tier (quadrant = Monitor) rather than elevating all rows to
    # Star via a >= comparison against 0.
    potential_hi: pd.Series = potential_values > potential_threshold
    actual_hi: pd.Series = actual_values > actual_threshold

    quadrant: pd.Series = pd.Series(
        [
            _classify_quadrant(bool(p), bool(a))
            for p, a in zip(potential_hi, actual_hi)
        ],
        index=work.index,
        dtype="object",
    )

    potential_tier: pd.Series = potential_hi.map({True: "High", False: "Low"})
    actual_tier: pd.Series = actual_hi.map({True: "High", False: "Low"})

    result: pd.DataFrame = work.assign(
        potential_norm=potential_norm.round(2).values,
        actual_norm=actual_norm.round(2).values,
        gap_score=gap_score.values,
        quadrant=quadrant.values,
        potential_tier=potential_tier.values,
        actual_tier=actual_tier.values,
    )

    logger.debug(
        "Classified %d HCPs (potential_split=%.2f, actual_split=%.2f).",
        len(result),
        potential_threshold,
        actual_threshold,
    )
    return result


def summarise_quadrants(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise HCP counts, average gap, and totals per quadrant.

    Args:
        matrix_df: Output of :func:`compute_potential_actual_matrix`.  Must
            contain the ``quadrant`` and ``gap_score`` columns.

    Returns:
        A new DataFrame with one row per observed quadrant and columns
        ``quadrant``, ``hcp_count``, ``avg_potential``, ``avg_actual``,
        ``avg_gap_score``, and ``pct_of_cohort`` (0-100, two decimals).
        Rows are ordered using :data:`QUADRANT_LABELS`.

    Raises:
        TypeError: If *matrix_df* is not a :class:`pandas.DataFrame`.
        ValueError: If *matrix_df* is empty or required columns are
            missing.
    """
    if not isinstance(matrix_df, pd.DataFrame):
        raise TypeError(
            f"matrix_df must be a pandas DataFrame, got {type(matrix_df).__name__}."
        )
    if matrix_df.empty:
        raise ValueError("matrix_df is empty; nothing to summarise.")
    required = {"quadrant", "gap_score", "potential_norm", "actual_norm"}
    missing = required - set(matrix_df.columns)
    if missing:
        raise ValueError(
            f"matrix_df is missing required columns: {sorted(missing)}. "
            "Run compute_potential_actual_matrix() first."
        )

    grouped: pd.DataFrame = (
        matrix_df.groupby("quadrant", as_index=False)
        .agg(
            hcp_count=("quadrant", "size"),
            avg_potential=("potential_norm", "mean"),
            avg_actual=("actual_norm", "mean"),
            avg_gap_score=("gap_score", "mean"),
        )
    )

    total: int = int(grouped["hcp_count"].sum())
    grouped["pct_of_cohort"] = (
        grouped["hcp_count"] / max(total, 1) * 100.0
    ).round(2)

    for num_col in ("avg_potential", "avg_actual", "avg_gap_score"):
        grouped[num_col] = grouped[num_col].round(2)

    # Preserve the canonical quadrant order for human-friendly reporting.
    order_index: Dict[str, int] = {
        name: idx for idx, name in enumerate(QUADRANT_LABELS)
    }
    grouped["_order"] = grouped["quadrant"].map(
        lambda q: order_index.get(q, len(QUADRANT_LABELS))
    )
    grouped = (
        grouped.sort_values("_order", kind="mergesort")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    return grouped


def top_growth_opportunities(
    matrix_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return the *n* HCPs with the largest untapped-demand gap.

    "Untapped demand" is a positive ``gap_score`` (high potential, low
    actual).  HCPs in the ``"Grow"`` quadrant are naturally concentrated
    here, but the ranking also surfaces ``"Star"`` HCPs whose actuals
    still trail their ceiling.

    Args:
        matrix_df: Output of :func:`compute_potential_actual_matrix`.
        n: Number of HCPs to return. Must be a positive integer.

    Returns:
        A new DataFrame with up to *n* rows, sorted by ``gap_score``
        descending.  When fewer than *n* HCPs have a positive gap, the
        result is truncated accordingly (never padded).

    Raises:
        TypeError: If *matrix_df* is not a :class:`pandas.DataFrame` or
            *n* is not an int.
        ValueError: If *n* is non-positive, *matrix_df* is empty, or
            ``gap_score`` is missing.
    """
    if not isinstance(matrix_df, pd.DataFrame):
        raise TypeError(
            f"matrix_df must be a pandas DataFrame, got {type(matrix_df).__name__}."
        )
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")
    if matrix_df.empty:
        raise ValueError("matrix_df is empty; nothing to rank.")
    if "gap_score" not in matrix_df.columns:
        raise ValueError(
            "matrix_df is missing required 'gap_score' column. "
            "Run compute_potential_actual_matrix() first."
        )

    positive: pd.DataFrame = matrix_df[matrix_df["gap_score"] > 0].copy()
    if positive.empty:
        return positive.head(0).reset_index(drop=True)

    return (
        positive.sort_values("gap_score", ascending=False, kind="mergesort")
        .head(n)
        .reset_index(drop=True)
    )

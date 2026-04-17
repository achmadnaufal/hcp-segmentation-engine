"""
RFM (Recency-Frequency-Monetary) scoring for healthcare professionals.

This module ranks HCPs on three commercially-meaningful dimensions of
prescribing behaviour:

- **Recency** — days since the HCP's last prescription. Lower is better.
- **Frequency** — number of prescriptions written in a recent window
  (e.g. 90 days). Higher is better.
- **Monetary** — total prescription dollar value in the same window.
  Higher is better.

Each dimension is converted into a 1-5 quintile score (5 = best) using
``pandas.qcut`` with rank-based tie handling. The three scores are then
combined into:

- A concatenated string code (e.g. ``"555"`` for top-tier HCPs).
- A composite numeric score in [0, 100] using configurable weights.
- A categorical RFM segment label (``"Champion"``, ``"Loyal"``,
  ``"At Risk"``, ``"Lost"``, or ``"Casual"``).

Every function is **immutable** — input DataFrames are never modified;
new objects are always returned.

Author: github.com/achmadnaufal
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default RFM score weights (must sum to 1.0).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "recency": 0.30,
    "frequency": 0.35,
    "monetary": 0.35,
}

#: Number of quintiles. RFM uses 5 by convention.
NUM_QUINTILES: int = 5

#: Default column names expected on input DataFrames.
DEFAULT_COLUMNS: Dict[str, str] = {
    "hcp_id": "hcp_id",
    "last_rx_date": "last_rx_date",
    "rx_count": "rx_count_90d",
    "rx_value": "total_rx_value_usd",
}

#: Segment label thresholds applied to the composite RFM score (0-100).
SEGMENT_THRESHOLDS: Dict[str, float] = {
    "champion": 80.0,
    "loyal": 60.0,
    "casual": 40.0,
    "at_risk": 20.0,
    # below 20 -> "Lost"
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quintile_score(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    """Convert *series* into integer quintile scores in ``[1, NUM_QUINTILES]``.

    Uses rank-based binning so duplicate values (and even fully constant
    series) map to a sensible default rather than raising. When
    *ascending* is ``True``, larger values receive higher scores; when
    ``False``, smaller values receive higher scores (used for recency,
    where fewer days since last Rx is better).

    Args:
        series: Numeric values to bin.
        ascending: If True, larger values get higher scores. If False,
            smaller values get higher scores.

    Returns:
        An ``int`` ``pandas.Series`` aligned with ``series.index``.
    """
    n_unique: int = series.nunique(dropna=True)
    if n_unique == 0:
        return pd.Series(
            [NUM_QUINTILES // 2 + 1] * len(series),
            index=series.index,
            dtype=int,
        )

    # If all values are identical (zero variance), assign the median
    # quintile so downstream weighting still produces a defined score.
    if n_unique == 1:
        return pd.Series(
            [NUM_QUINTILES // 2 + 1] * len(series),
            index=series.index,
            dtype=int,
        )

    ranked: pd.Series = series.rank(method="first", ascending=ascending)
    bins: int = min(NUM_QUINTILES, n_unique)
    quintile: pd.Series = pd.qcut(
        ranked,
        q=bins,
        labels=list(range(1, bins + 1)),
    )
    return quintile.astype(int)


def _validate_weights(weights: Dict[str, float]) -> None:
    """Raise ``ValueError`` if *weights* is malformed.

    Args:
        weights: Mapping with keys ``"recency"``, ``"frequency"``, and
            ``"monetary"``. Values must be non-negative and sum to 1.0
            (within a small tolerance).

    Raises:
        ValueError: When required keys are missing, weights are negative,
            or the weights do not sum to 1.0.
    """
    required = {"recency", "frequency", "monetary"}
    missing = required - set(weights)
    if missing:
        raise ValueError(
            f"weights is missing required keys: {sorted(missing)}."
        )
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative.")
    total: float = float(sum(weights[k] for k in required))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"weights for recency/frequency/monetary must sum to 1.0, "
            f"got {total:.4f}."
        )


def _label_segment(score: float) -> str:
    """Map a composite RFM score in ``[0, 100]`` to a segment label.

    Args:
        score: Composite RFM score.

    Returns:
        One of ``"Champion"``, ``"Loyal"``, ``"Casual"``, ``"At Risk"``,
        or ``"Lost"``.
    """
    if score >= SEGMENT_THRESHOLDS["champion"]:
        return "Champion"
    if score >= SEGMENT_THRESHOLDS["loyal"]:
        return "Loyal"
    if score >= SEGMENT_THRESHOLDS["casual"]:
        return "Casual"
    if score >= SEGMENT_THRESHOLDS["at_risk"]:
        return "At Risk"
    return "Lost"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rfm_scores(
    df: pd.DataFrame,
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    weights: Optional[Dict[str, float]] = None,
    hcp_id_col: str = DEFAULT_COLUMNS["hcp_id"],
    last_rx_date_col: str = DEFAULT_COLUMNS["last_rx_date"],
    rx_count_col: str = DEFAULT_COLUMNS["rx_count"],
    rx_value_col: str = DEFAULT_COLUMNS["rx_value"],
) -> pd.DataFrame:
    """Compute RFM scores and a composite segment label for each HCP.

    The function does not mutate *df*. NaN values in numeric columns are
    treated as zero, and missing ``last_rx_date`` values fall back to the
    *reference_date* (i.e. recency = 0 days, the most-recent bucket).

    Args:
        df: HCP-level DataFrame containing one row per HCP.
        reference_date: Date from which recency is measured. Defaults to
            ``pandas.Timestamp.today()`` when ``None``.
        weights: Optional override for the composite-score weights. Must
            contain keys ``"recency"``, ``"frequency"``, ``"monetary"``
            and sum to 1.0.
        hcp_id_col: Column holding HCP identifiers.
        last_rx_date_col: Column with the date of each HCP's last
            prescription (parseable by ``pd.to_datetime``).
        rx_count_col: Column with prescription counts in the analysis
            window.
        rx_value_col: Column with prescription dollar value in the
            analysis window.

    Returns:
        A new DataFrame with one row per HCP and the following added
        columns: ``recency_days``, ``r_score``, ``f_score``, ``m_score``,
        ``rfm_code``, ``rfm_score`` (composite, 0-100, rounded to 2 d.p.),
        and ``rfm_segment``.

    Raises:
        TypeError: If *df* is not a ``pandas.DataFrame``.
        ValueError: If *df* is empty, required columns are missing, or
            *weights* is malformed.

    Example:
        >>> import pandas as pd
        >>> from src.rfm_scorer import compute_rfm_scores
        >>> df = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3"],
        ...     "last_rx_date": ["2026-04-10", "2026-01-15", "2025-09-01"],
        ...     "rx_count_90d": [120, 40, 5],
        ...     "total_rx_value_usd": [45000.0, 12000.0, 800.0],
        ... })
        >>> result = compute_rfm_scores(df, reference_date="2026-04-18")
        >>> set(["r_score", "f_score", "m_score", "rfm_code",
        ...      "rfm_score", "rfm_segment"]).issubset(result.columns)
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise ValueError("df is empty. Provide at least one HCP record.")

    required = [hcp_id_col, last_rx_date_col, rx_count_col, rx_value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    effective_weights: Dict[str, float] = dict(weights or DEFAULT_WEIGHTS)
    _validate_weights(effective_weights)

    ref_ts: pd.Timestamp = (
        pd.Timestamp(reference_date)
        if reference_date is not None
        else pd.Timestamp.today().normalize()
    )

    work: pd.DataFrame = df.copy()

    last_rx: pd.Series = pd.to_datetime(
        work[last_rx_date_col], errors="coerce"
    )
    last_rx = last_rx.fillna(ref_ts)
    recency_days: pd.Series = (ref_ts - last_rx).dt.days.clip(lower=0)

    rx_count: pd.Series = (
        pd.to_numeric(work[rx_count_col], errors="coerce")
        .fillna(0)
        .astype(float)
    )
    rx_value: pd.Series = (
        pd.to_numeric(work[rx_value_col], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    r_score: pd.Series = _quintile_score(recency_days, ascending=False)
    f_score: pd.Series = _quintile_score(rx_count, ascending=True)
    m_score: pd.Series = _quintile_score(rx_value, ascending=True)

    rfm_code: pd.Series = (
        r_score.astype(str) + f_score.astype(str) + m_score.astype(str)
    )

    # Composite weighted score on a 0-100 scale: each quintile -> 0-100.
    def _to_pct(series: pd.Series) -> pd.Series:
        return (series - 1) / (NUM_QUINTILES - 1) * 100.0

    composite: pd.Series = (
        _to_pct(r_score) * effective_weights["recency"]
        + _to_pct(f_score) * effective_weights["frequency"]
        + _to_pct(m_score) * effective_weights["monetary"]
    ).round(2)

    segment: pd.Series = composite.map(_label_segment)

    result: pd.DataFrame = work.assign(
        recency_days=recency_days.astype(int).values,
        r_score=r_score.values,
        f_score=f_score.values,
        m_score=m_score.values,
        rfm_code=rfm_code.values,
        rfm_score=composite.values,
        rfm_segment=segment.values,
    )

    logger.debug(
        "Computed RFM for %d HCPs (reference_date=%s).",
        len(result),
        ref_ts.date(),
    )
    return result


def get_top_hcps(
    rfm_df: pd.DataFrame,
    n: int = 10,
    score_col: str = "rfm_score",
) -> pd.DataFrame:
    """Return the *n* HCPs with the highest composite RFM score.

    Args:
        rfm_df: Output of :func:`compute_rfm_scores`.
        n: Number of HCPs to return. Must be a positive integer.
        score_col: Column to sort on (default ``"rfm_score"``).

    Returns:
        A new DataFrame containing up to *n* rows, sorted by *score_col*
        in descending order.

    Raises:
        TypeError: If *rfm_df* is not a ``pandas.DataFrame`` or *n* is
            not an integer.
        ValueError: If *rfm_df* is empty, *n* is non-positive, or
            *score_col* is missing.
    """
    if not isinstance(rfm_df, pd.DataFrame):
        raise TypeError(
            f"rfm_df must be a pandas DataFrame, got {type(rfm_df).__name__}."
        )
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")
    if rfm_df.empty:
        raise ValueError("rfm_df is empty; nothing to rank.")
    if score_col not in rfm_df.columns:
        raise ValueError(
            f"score_col '{score_col}' not in rfm_df columns: "
            f"{list(rfm_df.columns)}"
        )

    return (
        rfm_df.sort_values(score_col, ascending=False, kind="mergesort")
        .head(n)
        .reset_index(drop=True)
    )


def summarise_rfm_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise HCP counts and average composite score per RFM segment.

    Args:
        rfm_df: Output of :func:`compute_rfm_scores`. Must contain
            ``rfm_segment`` and ``rfm_score`` columns.

    Returns:
        A new DataFrame with one row per segment and columns
        ``segment``, ``hcp_count``, ``avg_rfm_score``,
        sorted by ``avg_rfm_score`` descending.

    Raises:
        TypeError: If *rfm_df* is not a ``pandas.DataFrame``.
        ValueError: If required columns are missing or the DataFrame is
            empty.
    """
    if not isinstance(rfm_df, pd.DataFrame):
        raise TypeError(
            f"rfm_df must be a pandas DataFrame, got {type(rfm_df).__name__}."
        )
    if rfm_df.empty:
        raise ValueError("rfm_df is empty; cannot summarise.")
    needed = {"rfm_segment", "rfm_score"}
    missing = needed - set(rfm_df.columns)
    if missing:
        raise ValueError(
            f"rfm_df is missing required columns: {sorted(missing)}."
        )

    grouped = (
        rfm_df.groupby("rfm_segment", as_index=False)
        .agg(
            hcp_count=("rfm_segment", "size"),
            avg_rfm_score=("rfm_score", "mean"),
        )
        .rename(columns={"rfm_segment": "segment"})
        .sort_values("avg_rfm_score", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["avg_rfm_score"] = grouped["avg_rfm_score"].round(2)
    return grouped

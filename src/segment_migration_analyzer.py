"""
Segment migration analysis for HCP cohorts across two time periods.

Given two DataFrames of segmented HCP data (period A and period B), this
module computes:

- A segment-to-segment migration matrix (counts and percentages).
- Per-HCP transition records with upgrade / downgrade / stable flags.
- A churn-risk score for HCPs whose segment has degraded.

Every function is **immutable**: inputs are never modified; new objects are
always returned.

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

#: Ordered segment hierarchy — higher index = higher value segment.
SEGMENT_ORDER: List[str] = [
    "Dormant",
    "Low Activity",
    "Standard",
    "Digital Adopter",
    "Growth Target",
    "High-Value KOL",
]

#: Default column name for HCP identifier.
DEFAULT_HCP_ID_COL: str = "hcp_id"

#: Default column name for computed segment.
DEFAULT_SEGMENT_COL: str = "computed_segment"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _segment_rank(segment: str) -> int:
    """Return the numeric rank of *segment* in :data:`SEGMENT_ORDER`.

    Unrecognised segment names receive rank ``-1`` so they sort below all
    known segments.

    Args:
        segment: Segment name string.

    Returns:
        Integer rank (0–5 for known segments, -1 for unknown).
    """
    try:
        return SEGMENT_ORDER.index(segment)
    except ValueError:
        return -1


def _validate_migration_inputs(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    hcp_id_col: str,
    segment_col: str,
) -> None:
    """Validate the two input DataFrames for migration analysis.

    Args:
        df_before: DataFrame for the earlier period.
        df_after: DataFrame for the later period.
        hcp_id_col: Name of the HCP identifier column.
        segment_col: Name of the segment column.

    Raises:
        TypeError: If either argument is not a :class:`pandas.DataFrame`.
        ValueError: If either DataFrame is empty, or required columns are
            absent.
    """
    for label, df in (("df_before", df_before), ("df_after", df_after)):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{label} must be a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            raise ValueError(
                f"{label} is empty. Provide at least one HCP record."
            )
        missing_cols: List[str] = [
            c for c in (hcp_id_col, segment_col) if c not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"{label} is missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
    if not hcp_id_col or not hcp_id_col.strip():
        raise ValueError("hcp_id_col must be a non-empty string.")
    if not segment_col or not segment_col.strip():
        raise ValueError("segment_col must be a non-empty string.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_migration_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    hcp_id_col: str = DEFAULT_HCP_ID_COL,
    segment_col: str = DEFAULT_SEGMENT_COL,
) -> pd.DataFrame:
    """Build a per-HCP transition table across two segmentation periods.

    Only HCPs that appear in **both** DataFrames are included (inner join on
    *hcp_id_col*).  HCPs present in only one period are silently excluded; a
    count of unmatched HCPs is logged at DEBUG level.

    Each row in the result represents one HCP and contains:

    - The before and after segment labels.
    - A numeric rank for each period (based on :data:`SEGMENT_ORDER`).
    - A ``direction`` flag: ``"upgrade"``, ``"downgrade"``, or ``"stable"``.
    - A boolean ``is_churned`` flag (True when the HCP moved to ``"Dormant"``
      from a higher segment).
    - A float ``churn_risk_score`` in [0, 1]: the normalised rank drop
      (0 for stable / upgrades, larger values for bigger drops).

    Args:
        df_before: HCP DataFrame for period A (must contain *hcp_id_col* and
            *segment_col*).
        df_after: HCP DataFrame for period B (must contain *hcp_id_col* and
            *segment_col*).
        hcp_id_col: Name of the HCP identifier column (default ``"hcp_id"``).
        segment_col: Name of the segment column (default
            ``"computed_segment"``).

    Returns:
        A new DataFrame with one row per matched HCP and columns:

        - ``hcp_id`` -- HCP identifier.
        - ``segment_before`` -- segment in period A.
        - ``segment_after`` -- segment in period B.
        - ``rank_before`` -- numeric rank of ``segment_before``.
        - ``rank_after`` -- numeric rank of ``segment_after``.
        - ``rank_delta`` -- ``rank_after - rank_before`` (positive = upgrade).
        - ``direction`` -- ``"upgrade"``, ``"downgrade"``, or ``"stable"``.
        - ``is_churned`` -- True if HCP is now ``"Dormant"``.
        - ``churn_risk_score`` -- normalised rank drop in [0, 1].

    Raises:
        TypeError: If either *df_before* or *df_after* is not a DataFrame.
        ValueError: If either DataFrame is empty or required columns are
            missing.

    Example:
        >>> import pandas as pd
        >>> from src.segment_migration_analyzer import compute_migration_table
        >>> before = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3"],
        ...     "computed_segment": ["Standard", "Growth Target", "Dormant"],
        ... })
        >>> after = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3"],
        ...     "computed_segment": ["Growth Target", "Standard", "Low Activity"],
        ... })
        >>> result = compute_migration_table(before, after)
        >>> result[["hcp_id", "direction", "rank_delta"]].to_string(index=False)
        ' hcp_id  direction  rank_delta\\n      H1    upgrade           2\\n      H2  downgrade          -2\\n      H3    upgrade           1'
    """
    _validate_migration_inputs(df_before, df_after, hcp_id_col, segment_col)

    before_slim: pd.DataFrame = df_before[[hcp_id_col, segment_col]].copy()
    after_slim: pd.DataFrame = df_after[[hcp_id_col, segment_col]].copy()

    merged: pd.DataFrame = before_slim.merge(
        after_slim,
        on=hcp_id_col,
        how="inner",
        suffixes=("_before", "_after"),
    )

    n_before = len(before_slim)
    n_after = len(after_slim)
    n_matched = len(merged)
    logger.debug(
        "Migration join: %d before, %d after, %d matched HCPs.",
        n_before,
        n_after,
        n_matched,
    )

    seg_before_col = f"{segment_col}_before"
    seg_after_col = f"{segment_col}_after"

    rank_before: pd.Series = merged[seg_before_col].map(_segment_rank)
    rank_after: pd.Series = merged[seg_after_col].map(_segment_rank)
    rank_delta: pd.Series = rank_after - rank_before

    direction: pd.Series = pd.Series("stable", index=merged.index, dtype=str)
    direction = direction.where(rank_delta <= 0, other="upgrade")
    direction = direction.where(rank_delta >= 0, other="downgrade")

    is_churned: pd.Series = merged[seg_after_col] == "Dormant"

    max_rank: int = max(len(SEGMENT_ORDER) - 1, 1)
    churn_risk_score: pd.Series = (
        (-rank_delta).clip(lower=0) / max_rank
    ).round(4)

    result: pd.DataFrame = pd.DataFrame(
        {
            "hcp_id": merged[hcp_id_col].values,
            "segment_before": merged[seg_before_col].values,
            "segment_after": merged[seg_after_col].values,
            "rank_before": rank_before.values,
            "rank_after": rank_after.values,
            "rank_delta": rank_delta.values,
            "direction": direction.values,
            "is_churned": is_churned.values,
            "churn_risk_score": churn_risk_score.values,
        },
        index=range(n_matched),
    )

    return result


def build_migration_matrix(
    migration_table: pd.DataFrame,
    normalise: bool = False,
) -> pd.DataFrame:
    """Produce a segment-to-segment migration matrix from a migration table.

    Rows represent the segment in period A (before), columns represent the
    segment in period B (after).  Only segments present in the data appear as
    labels; the order follows :data:`SEGMENT_ORDER` for known names.

    Args:
        migration_table: Output of :func:`compute_migration_table` (or any
            DataFrame with ``segment_before`` and ``segment_after`` columns).
        normalise: When ``True``, divide each row by its row total so values
            represent the fraction of HCPs that moved to each segment.  Rows
            with zero total are left as zeros.

    Returns:
        A new DataFrame (counts or fractions) whose index and columns are
        segment labels ordered by :data:`SEGMENT_ORDER`.

    Raises:
        TypeError: If *migration_table* is not a :class:`pandas.DataFrame`.
        ValueError: If *migration_table* is empty, or if the required
            ``segment_before`` / ``segment_after`` columns are absent.

    Example:
        >>> import pandas as pd
        >>> from src.segment_migration_analyzer import (
        ...     compute_migration_table, build_migration_matrix
        ... )
        >>> before = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2"],
        ...     "computed_segment": ["Standard", "Standard"],
        ... })
        >>> after = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2"],
        ...     "computed_segment": ["Growth Target", "Standard"],
        ... })
        >>> tbl = compute_migration_table(before, after)
        >>> mx = build_migration_matrix(tbl)
        >>> mx.loc["Standard", "Growth Target"]
        1
    """
    if not isinstance(migration_table, pd.DataFrame):
        raise TypeError(
            "migration_table must be a pandas DataFrame, "
            f"got {type(migration_table).__name__}."
        )
    if migration_table.empty:
        raise ValueError("migration_table is empty; cannot build matrix.")
    required = {"segment_before", "segment_after"}
    missing = required - set(migration_table.columns)
    if missing:
        raise ValueError(
            f"migration_table is missing columns: {sorted(missing)}. "
            f"Available columns: {list(migration_table.columns)}"
        )

    matrix: pd.DataFrame = pd.crosstab(
        migration_table["segment_before"],
        migration_table["segment_after"],
    )

    # Re-index to standard segment order, keeping only labels present in data.
    all_labels: List[str] = migration_table["segment_before"].tolist() + \
        migration_table["segment_after"].tolist()
    present: List[str] = [s for s in SEGMENT_ORDER if s in set(all_labels)]

    matrix = matrix.reindex(index=present, columns=present, fill_value=0)

    if normalise:
        row_totals: pd.Series = matrix.sum(axis=1)
        row_totals_safe: pd.Series = row_totals.replace(0, np.nan)
        matrix = matrix.div(row_totals_safe, axis=0).fillna(0.0).round(4)

    return matrix


def summarise_migrations(
    migration_table: pd.DataFrame,
) -> Dict[str, object]:
    """Return high-level migration statistics for a cohort.

    Args:
        migration_table: Output of :func:`compute_migration_table`.

    Returns:
        A plain dict (immutable values) with keys:

        - ``total_hcps`` (int) -- total matched HCPs.
        - ``upgraded`` (int) -- HCPs whose segment improved.
        - ``downgraded`` (int) -- HCPs whose segment declined.
        - ``stable`` (int) -- HCPs in the same segment both periods.
        - ``churned`` (int) -- HCPs now in ``"Dormant"``.
        - ``pct_upgraded`` (float) -- percentage upgraded, rounded to 1 d.p.
        - ``pct_downgraded`` (float) -- percentage downgraded, rounded to 1 d.p.
        - ``pct_churned`` (float) -- percentage churned, rounded to 1 d.p.
        - ``avg_churn_risk`` (float) -- mean churn risk score, rounded to 4 d.p.
        - ``top_churn_risk_hcps`` (list[str]) -- up to 5 HCP IDs with highest
          churn risk, ordered descending.

    Raises:
        TypeError: If *migration_table* is not a :class:`pandas.DataFrame`.
        ValueError: If *migration_table* is empty or is missing required
            columns.

    Example:
        >>> import pandas as pd
        >>> from src.segment_migration_analyzer import (
        ...     compute_migration_table, summarise_migrations
        ... )
        >>> before = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3"],
        ...     "computed_segment": ["Standard", "Growth Target", "Low Activity"],
        ... })
        >>> after = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2", "H3"],
        ...     "computed_segment": ["Growth Target", "Dormant", "Low Activity"],
        ... })
        >>> tbl = compute_migration_table(before, after)
        >>> stats = summarise_migrations(tbl)
        >>> stats["upgraded"]
        1
        >>> stats["churned"]
        1
    """
    if not isinstance(migration_table, pd.DataFrame):
        raise TypeError(
            "migration_table must be a pandas DataFrame, "
            f"got {type(migration_table).__name__}."
        )
    if migration_table.empty:
        raise ValueError(
            "migration_table is empty. Run compute_migration_table() first."
        )
    required_cols = {"direction", "is_churned", "churn_risk_score", "hcp_id"}
    missing = required_cols - set(migration_table.columns)
    if missing:
        raise ValueError(
            f"migration_table is missing columns: {sorted(missing)}. "
            "Ensure it was produced by compute_migration_table()."
        )

    total: int = len(migration_table)
    upgraded: int = int((migration_table["direction"] == "upgrade").sum())
    downgraded: int = int((migration_table["direction"] == "downgrade").sum())
    stable: int = int((migration_table["direction"] == "stable").sum())
    churned: int = int(migration_table["is_churned"].sum())

    def _pct(n: int) -> float:
        return round(n / total * 100, 1) if total > 0 else 0.0

    avg_churn_risk: float = round(
        float(migration_table["churn_risk_score"].mean()), 4
    )

    top_churn: List[str] = (
        migration_table
        .nlargest(5, "churn_risk_score")["hcp_id"]
        .tolist()
    )

    return {
        "total_hcps": total,
        "upgraded": upgraded,
        "downgraded": downgraded,
        "stable": stable,
        "churned": churned,
        "pct_upgraded": _pct(upgraded),
        "pct_downgraded": _pct(downgraded),
        "pct_churned": _pct(churned),
        "avg_churn_risk": avg_churn_risk,
        "top_churn_risk_hcps": top_churn,
    }

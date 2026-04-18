"""
Calling-plan allocator for HCP sales-force deployment.

Given a segmented HCP cohort (the output of
:class:`src.main.HCPSegmentationEngine.run_full_pipeline` or any DataFrame
with ``computed_segment`` and ``composite_score`` columns), this module
distributes a fixed annual call-capacity budget across HCPs using a
segment-weighted, priority-driven allocation algorithm.

Key design decisions:

- **Immutable** — inputs are never modified; every function returns a new
  DataFrame or dict.
- **Deterministic** — the same inputs always produce the same allocation;
  ties are broken by descending ``composite_score`` and then ascending
  ``hcp_id`` so results are reproducible.
- **Robust** — handles empty cohorts, zero-budget scenarios, missing
  optional columns, identical scores, and NaN values without raising.

Public API:

- :func:`calculate_priority_score` — blend composite and segment weight
  into a single 0-100 priority score per HCP.
- :func:`allocate_calls` — distribute a total-call budget across HCPs
  honouring per-HCP minimum and maximum caps.
- :func:`summarise_allocation` — per-segment allocation summary (calls,
  HCPs, coverage pct, average priority).

Author: github.com/achmadnaufal
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default per-segment weight (0.0-1.0) used when blending composite_score
#: with segment priority in :func:`calculate_priority_score`.  Higher weight
#: means the segment is more strategically important.
DEFAULT_SEGMENT_WEIGHTS: Dict[str, float] = {
    "High-Value KOL": 1.00,
    "Growth Target": 0.85,
    "Digital Adopter": 0.65,
    "Standard": 0.45,
    "Low Activity": 0.20,
    "Dormant": 0.05,
}

#: Default per-segment target calls-per-HCP used by :func:`allocate_calls`
#: as an initial ideal cadence before budget scaling and capping.
DEFAULT_TARGET_CALLS_PER_SEGMENT: Dict[str, int] = {
    "High-Value KOL": 24,   # ~2 per month
    "Growth Target": 18,    # ~1.5 per month
    "Digital Adopter": 12,  # 1 per month
    "Standard": 8,          # ~quarterly+
    "Low Activity": 4,      # quarterly
    "Dormant": 0,           # excluded by default
}

#: Default minimum / maximum per-HCP annual call caps applied to every
#: allocation.  Protects against over- or under-servicing a single HCP.
DEFAULT_MIN_CALLS_PER_HCP: int = 0
DEFAULT_MAX_CALLS_PER_HCP: int = 52  # hard cap = one call per week

#: Weight blend between composite_score (0-100) and segment weight (0-1).
#: A value of 0.6 means 60% composite + 40% segment contribution.
DEFAULT_COMPOSITE_BLEND: float = 0.6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_dataframe(df: pd.DataFrame, name: str) -> None:
    """Raise :class:`TypeError` unless *df* is a ``pandas.DataFrame``.

    Args:
        df: Candidate object to check.
        name: Parameter name used in the error message.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}."
        )


def _require_columns(df: pd.DataFrame, columns: List[str], name: str) -> None:
    """Raise :class:`ValueError` unless *df* contains every column in *columns*.

    Args:
        df: DataFrame to check.
        columns: Required column names.
        name: Parameter name used in the error message.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _clip_to_bounds(values: pd.Series, low: int, high: int) -> pd.Series:
    """Return a new Series with each value clamped to ``[low, high]``.

    Args:
        values: Integer or float Series to clip.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        A new Series of the same dtype with values constrained to the
        given range.  The input is not mutated.
    """
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) cannot exceed upper bound ({high})."
        )
    return values.clip(lower=low, upper=high)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_priority_score(
    df: pd.DataFrame,
    segment_col: str = "computed_segment",
    composite_col: str = "composite_score",
    segment_weights: Optional[Mapping[str, float]] = None,
    composite_blend: float = DEFAULT_COMPOSITE_BLEND,
) -> pd.DataFrame:
    """Compute a single 0-100 ``priority_score`` per HCP.

    The priority score blends two signals:

    1. **Composite score** (``composite_col``) — already on 0-100 scale,
       produced by the segmentation engine.
    2. **Segment weight** (``segment_weights``) — strategic multiplier
       per segment; the default favours KOLs and Growth Targets.

    The blend is::

        priority = composite * blend + segment_weight * 100 * (1 - blend)

    Unknown segment names fall back to weight ``0.0``.  NaN composite
    scores are treated as zero.

    Args:
        df: HCP DataFrame with at least ``segment_col`` and
            ``composite_col``.
        segment_col: Column holding segment labels.  Defaults to
            ``"computed_segment"``.
        composite_col: Column holding composite scores (0-100).  Defaults
            to ``"composite_score"``.
        segment_weights: Optional override for
            :data:`DEFAULT_SEGMENT_WEIGHTS`.  Values should be in
            ``[0.0, 1.0]``.
        composite_blend: Weight given to *composite_col* in the final
            blend (remainder flows to *segment_weights*).  Must be in
            ``[0.0, 1.0]``.

    Returns:
        A new DataFrame with an added ``priority_score`` column (float,
        rounded to 2 d.p.).  The input is not modified.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.
        ValueError: If required columns are absent or *composite_blend*
            is outside ``[0.0, 1.0]``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "hcp_id": ["H1", "H2"],
        ...     "computed_segment": ["High-Value KOL", "Dormant"],
        ...     "composite_score": [85.0, 5.0],
        ... })
        >>> result = calculate_priority_score(df)
        >>> result.loc[0, "priority_score"] > result.loc[1, "priority_score"]
        True
    """
    _require_dataframe(df, "df")
    _require_columns(df, [segment_col, composite_col], "df")
    if not 0.0 <= composite_blend <= 1.0:
        raise ValueError(
            f"composite_blend must be in [0.0, 1.0], got {composite_blend}."
        )

    weights: Dict[str, float] = dict(segment_weights or DEFAULT_SEGMENT_WEIGHTS)
    for seg, w in weights.items():
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(
                f"segment_weights['{seg}'] must be in [0.0, 1.0], got {w}."
            )

    result: pd.DataFrame = df.copy()

    composite: pd.Series = (
        pd.to_numeric(result[composite_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
    )
    segment_weight: pd.Series = (
        result[segment_col].map(weights).fillna(0.0).astype(float)
    )

    priority: pd.Series = (
        composite * composite_blend
        + segment_weight * 100.0 * (1.0 - composite_blend)
    ).round(2)

    result = result.copy()
    result["priority_score"] = priority
    return result


def allocate_calls(
    df: pd.DataFrame,
    total_calls_budget: int,
    segment_col: str = "computed_segment",
    composite_col: str = "composite_score",
    target_calls_per_segment: Optional[Mapping[str, int]] = None,
    min_calls_per_hcp: int = DEFAULT_MIN_CALLS_PER_HCP,
    max_calls_per_hcp: int = DEFAULT_MAX_CALLS_PER_HCP,
    segment_weights: Optional[Mapping[str, float]] = None,
    composite_blend: float = DEFAULT_COMPOSITE_BLEND,
) -> pd.DataFrame:
    """Distribute *total_calls_budget* across HCPs based on priority.

    Algorithm:

    1. Compute a ``priority_score`` (via :func:`calculate_priority_score`).
    2. Seed each HCP with their segment's *target_calls_per_segment* value
       (zero when the segment is not listed).
    3. Scale all seeds proportionally so the sum equals
       ``total_calls_budget``.
    4. Nudge the scaled values by each HCP's priority share so higher
       priority HCPs receive a larger slice within their segment.
    5. Clip each value to ``[min_calls_per_hcp, max_calls_per_hcp]``.
    6. Round to integers and settle any ±1 rounding drift against the
       highest-priority HCPs so the total matches the budget exactly
       (bounded by per-HCP caps).

    Edge cases:

    - **Empty DataFrame** — returns a copy with an added ``allocated_calls``
      column of dtype int.
    - **Zero budget** — every HCP receives ``min_calls_per_hcp``.
    - **All identical priorities** — calls are distributed as evenly as
      possible subject to per-HCP caps.
    - **Budget exceeds per-HCP caps × HCPs** — allocation saturates at the
      maximum-per-HCP cap and the excess budget is dropped (logged at
      WARNING level).

    Args:
        df: HCP DataFrame with at least ``segment_col`` and
            ``composite_col``.
        total_calls_budget: Total annual call budget (non-negative int).
        segment_col: Column holding segment labels.
        composite_col: Column holding composite scores (0-100).
        target_calls_per_segment: Optional override for
            :data:`DEFAULT_TARGET_CALLS_PER_SEGMENT`.
        min_calls_per_hcp: Minimum calls any HCP can receive (>= 0).
        max_calls_per_hcp: Maximum calls any HCP can receive
            (>= *min_calls_per_hcp*).
        segment_weights: Optional override for
            :data:`DEFAULT_SEGMENT_WEIGHTS`.
        composite_blend: Composite / segment blend factor (0.0-1.0).

    Returns:
        A new DataFrame with two added columns:

        - ``priority_score`` (float) — as returned by
          :func:`calculate_priority_score`.
        - ``allocated_calls`` (int) — non-negative, within the per-HCP
          caps, summing to ``total_calls_budget`` where feasible.

        The input DataFrame is not modified.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.
        ValueError: If *total_calls_budget* is negative, min/max call
            caps are inconsistent, or required columns are absent.
    """
    _require_dataframe(df, "df")
    if not isinstance(total_calls_budget, (int, np.integer)) or isinstance(
        total_calls_budget, bool
    ):
        raise TypeError(
            f"total_calls_budget must be an int, "
            f"got {type(total_calls_budget).__name__}."
        )
    if total_calls_budget < 0:
        raise ValueError(
            f"total_calls_budget must be non-negative, got {total_calls_budget}."
        )
    if min_calls_per_hcp < 0:
        raise ValueError(
            f"min_calls_per_hcp must be non-negative, got {min_calls_per_hcp}."
        )
    if max_calls_per_hcp < min_calls_per_hcp:
        raise ValueError(
            f"max_calls_per_hcp ({max_calls_per_hcp}) must be >= "
            f"min_calls_per_hcp ({min_calls_per_hcp})."
        )

    if df.empty:
        empty_result: pd.DataFrame = df.copy()
        empty_result["priority_score"] = pd.Series(dtype=float)
        empty_result["allocated_calls"] = pd.Series(dtype=int)
        return empty_result

    _require_columns(df, [segment_col, composite_col], "df")

    targets: Dict[str, int] = dict(
        target_calls_per_segment or DEFAULT_TARGET_CALLS_PER_SEGMENT
    )

    scored: pd.DataFrame = calculate_priority_score(
        df,
        segment_col=segment_col,
        composite_col=composite_col,
        segment_weights=segment_weights,
        composite_blend=composite_blend,
    )

    n_hcp: int = len(scored)
    seed: pd.Series = (
        scored[segment_col].map(targets).fillna(0).astype(float)
    )

    # If all seeds are zero, fall back to an equal seed so priority can
    # still drive differentiation.
    if seed.sum() == 0:
        seed = pd.Series(1.0, index=scored.index)

    seed_total: float = float(seed.sum())
    scale: float = float(total_calls_budget) / seed_total if seed_total > 0 else 0.0
    scaled: pd.Series = seed * scale

    # Add a priority nudge within each segment: redistribute up to 20% of
    # each segment's allocation according to priority share.
    nudge_fraction: float = 0.20
    priority: pd.Series = scored["priority_score"].fillna(0.0)
    # Avoid division by zero for segments with zero priority total.
    segment_priority_total: pd.Series = priority.groupby(
        scored[segment_col]
    ).transform("sum").replace(0.0, np.nan)
    priority_share: pd.Series = (
        (priority / segment_priority_total).fillna(1.0 / max(n_hcp, 1))
    )
    segment_mean_share: pd.Series = priority_share.groupby(
        scored[segment_col]
    ).transform("mean")
    nudge: pd.Series = scaled * nudge_fraction * (priority_share - segment_mean_share)
    allocation: pd.Series = (scaled + nudge).clip(lower=0.0)

    # Apply per-HCP caps.
    allocation = _clip_to_bounds(
        allocation, low=float(min_calls_per_hcp), high=float(max_calls_per_hcp)
    )

    # Round to int and reconcile rounding drift against the budget.
    rounded: pd.Series = allocation.round().astype(int)
    max_feasible: int = int(max_calls_per_hcp * n_hcp)
    effective_budget: int = min(total_calls_budget, max_feasible)
    drift: int = int(effective_budget - rounded.sum())

    if effective_budget < total_calls_budget:
        logger.warning(
            "Budget %d exceeds per-HCP capacity %d; %d calls dropped.",
            total_calls_budget,
            max_feasible,
            total_calls_budget - max_feasible,
        )

    if drift != 0:
        # Deterministic ordering: highest priority first; tie-break by hcp_id.
        order_df = pd.DataFrame(
            {
                "priority_score": priority.values,
                "hcp_id": (
                    scored["hcp_id"].values
                    if "hcp_id" in scored.columns
                    else scored.index.astype(str).values
                ),
                "idx": scored.index.values,
            }
        )
        if drift > 0:
            ordered = order_df.sort_values(
                ["priority_score", "hcp_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            step = 1
            cap = max_calls_per_hcp
        else:
            ordered = order_df.sort_values(
                ["priority_score", "hcp_id"],
                ascending=[True, True],
                kind="mergesort",
            )
            step = -1
            cap = min_calls_per_hcp

        remaining = abs(drift)
        for idx in ordered["idx"].tolist():
            if remaining == 0:
                break
            current = rounded.at[idx]
            if step > 0 and current < cap:
                rounded.at[idx] = current + 1
                remaining -= 1
            elif step < 0 and current > cap:
                rounded.at[idx] = current - 1
                remaining -= 1
        if remaining != 0:
            logger.warning(
                "Could not reconcile allocation drift of %d calls "
                "(likely hit per-HCP caps).",
                remaining * step,
            )

    result: pd.DataFrame = scored.copy()
    result["allocated_calls"] = rounded.astype(int).values
    return result


def summarise_allocation(
    allocation_df: pd.DataFrame,
    segment_col: str = "computed_segment",
    calls_col: str = "allocated_calls",
    priority_col: str = "priority_score",
) -> pd.DataFrame:
    """Return a per-segment summary of a calling-plan allocation.

    Args:
        allocation_df: Output of :func:`allocate_calls` (or any DataFrame
            with the required columns).
        segment_col: Column holding segment labels.
        calls_col: Column with allocated calls per HCP.
        priority_col: Column with priority scores per HCP.

    Returns:
        A new DataFrame with one row per segment and columns:

        - ``segment`` (str)
        - ``hcp_count`` (int) — number of HCPs in the segment.
        - ``total_calls`` (int) — sum of allocated calls.
        - ``avg_calls_per_hcp`` (float) — mean calls, rounded to 2 d.p.
        - ``avg_priority`` (float) — mean priority, rounded to 2 d.p.
        - ``pct_of_budget`` (float) — share of total calls, rounded to
          1 d.p.

        Sorted by ``total_calls`` descending.

    Raises:
        TypeError: If *allocation_df* is not a DataFrame.
        ValueError: If required columns are missing or the DataFrame is
            empty.
    """
    _require_dataframe(allocation_df, "allocation_df")
    if allocation_df.empty:
        raise ValueError("allocation_df is empty; nothing to summarise.")
    _require_columns(
        allocation_df,
        [segment_col, calls_col, priority_col],
        "allocation_df",
    )

    grand_total: int = int(allocation_df[calls_col].sum())
    grand_total_safe: float = float(grand_total) if grand_total > 0 else 1.0

    grouped = (
        allocation_df.groupby(segment_col, as_index=False)
        .agg(
            hcp_count=(segment_col, "size"),
            total_calls=(calls_col, "sum"),
            avg_calls_per_hcp=(calls_col, "mean"),
            avg_priority=(priority_col, "mean"),
        )
        .rename(columns={segment_col: "segment"})
    )
    grouped["avg_calls_per_hcp"] = grouped["avg_calls_per_hcp"].round(2)
    grouped["avg_priority"] = grouped["avg_priority"].round(2)
    grouped["pct_of_budget"] = (
        grouped["total_calls"] / grand_total_safe * 100.0
    ).round(1)
    grouped["total_calls"] = grouped["total_calls"].astype(int)
    grouped["hcp_count"] = grouped["hcp_count"].astype(int)

    return grouped.sort_values(
        "total_calls", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

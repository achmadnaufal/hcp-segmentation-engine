"""
Decile-based HCP tiering (ABCDE) for pharmaceutical sales targeting.

This module assigns every Healthcare Professional (HCP) to one of ten
prescribing deciles (1 = highest Rx volume, 10 = lowest) and then maps
the deciles to an industry-standard A/B/C/D/E letter tier used by most
pharma commercial operations teams for call-plan prioritisation.

Decile-to-letter mapping (standard pharma convention):
    - A: Deciles 1-2   (top 20%, highest-prescribing HCPs)
    - B: Deciles 3-4   (next 20%)
    - C: Deciles 5-6   (middle 20%)
    - D: Deciles 7-8   (next 20%)
    - E: Deciles 9-10  (bottom 20%, lowest-prescribing)

The module also computes per-letter summary statistics (HCP count,
total Rx, share of total Rx) which are commonly used to justify field
force sizing and call-frequency decisions.

Author: github.com/achmadnaufal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RX_COLUMN: str = "prescriptions_last_12m"
DEFAULT_ID_COLUMN: str = "hcp_id"

# Decile index (1..10) -> letter tier
DECILE_TO_LETTER: Dict[int, str] = {
    1: "A", 2: "A",
    3: "B", 4: "B",
    5: "C", 6: "C",
    7: "D", 8: "D",
    9: "E", 10: "E",
}

LETTER_ORDER: List[str] = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LetterSummary:
    """Aggregate statistics for a single letter tier (A/B/C/D/E)."""

    letter: str
    hcp_count: int
    total_rx: float
    mean_rx: float
    min_rx: float
    max_rx: float
    rx_share_pct: float


@dataclass(frozen=True)
class DecileReport:
    """Full result of a decile-tiering run.

    Attributes:
        data: DataFrame with ``rx_decile`` and ``letter_tier`` appended.
        summaries: Per-letter aggregate statistics ordered A..E.
        total_rx: Sum of Rx across the whole input dataset.
        total_hcps: Number of HCPs in the input dataset.
    """

    data: pd.DataFrame
    summaries: List[LetterSummary] = field(default_factory=list)
    total_rx: float = 0.0
    total_hcps: int = 0

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame view of :attr:`summaries`."""
        rows = [
            {
                "letter": s.letter,
                "hcp_count": s.hcp_count,
                "total_rx": s.total_rx,
                "mean_rx": s.mean_rx,
                "min_rx": s.min_rx,
                "max_rx": s.max_rx,
                "rx_share_pct": s.rx_share_pct,
            }
            for s in self.summaries
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class DecileTieringEngine:
    """Assign ABCDE letter tiers to HCPs using prescribing deciles.

    Example:
        >>> engine = DecileTieringEngine()
        >>> df = pd.DataFrame({
        ...     "hcp_id": [f"HCP{i}" for i in range(10)],
        ...     "prescriptions_last_12m": [500, 450, 400, 350, 300,
        ...                                  250, 200, 150, 100, 50],
        ... })
        >>> report = engine.run(df)
        >>> report.data["letter_tier"].tolist()
        ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
    """

    def __init__(
        self,
        rx_column: str = DEFAULT_RX_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
    ) -> None:
        """Initialise the engine.

        Args:
            rx_column: Column containing the prescription volume metric.
            id_column: Column uniquely identifying each HCP.
        """
        if not rx_column or not isinstance(rx_column, str):
            raise ValueError("rx_column must be a non-empty string.")
        if not id_column or not isinstance(id_column, str):
            raise ValueError("id_column must be a non-empty string.")
        self._rx_column = rx_column
        self._id_column = id_column

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that *df* has the columns and rows needed for tiering.

        Args:
            df: Input DataFrame.

        Returns:
            ``True`` when validation passes.

        Raises:
            ValueError: If the dataset is empty, the required columns are
                missing, or the Rx column contains non-numeric data.
        """
        if df is None or df.empty:
            raise ValueError("Input dataset is empty; need at least one HCP row.")
        if self._rx_column not in df.columns:
            raise ValueError(
                f"Required Rx column '{self._rx_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if self._id_column not in df.columns:
            raise ValueError(
                f"Required ID column '{self._id_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if not pd.api.types.is_numeric_dtype(df[self._rx_column]):
            raise ValueError(
                f"Column '{self._rx_column}' must be numeric; "
                f"got dtype {df[self._rx_column].dtype}."
            )
        return True

    # ------------------------------------------------------------------
    # Decile assignment
    # ------------------------------------------------------------------

    def assign_deciles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append an ``rx_decile`` column (1..10) to a copy of *df*.

        Decile 1 = highest Rx volume, decile 10 = lowest.  Ties are broken
        with the 'first' ranking method so that every decile receives a
        consistent number of HCPs regardless of duplicate Rx values.

        Args:
            df: DataFrame containing the Rx column.

        Returns:
            A new DataFrame with ``rx_decile`` appended.  The original is
            not mutated.
        """
        self.validate(df)
        result = df.copy()

        rx_clean = result[self._rx_column].fillna(0.0).astype(float)
        n = len(result)

        if n == 0:
            result["rx_decile"] = pd.Series(dtype=int)
            return result

        # Rank descending so rank 1 = highest Rx.  'first' breaks ties by
        # input order, which keeps decile sizes as uniform as possible.
        ranks = rx_clean.rank(method="first", ascending=False)

        # Convert ranks into a 1..10 decile index using floor((r-1) * 10 / n).
        # This yields decile 1 for the top HCP and decile 10 for the bottom,
        # works correctly for n < 10, and distributes HCPs evenly for n >= 10.
        decile = np.floor((ranks - 1) * 10.0 / n).astype(int) + 1
        # Guarantee decile is within [1, 10] even in degenerate cases.
        decile = decile.clip(lower=1, upper=10)

        result["rx_decile"] = decile.astype(int)
        return result

    # ------------------------------------------------------------------
    # Letter mapping
    # ------------------------------------------------------------------

    def assign_letter_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``letter_tier`` column (A..E) based on ``rx_decile``.

        Args:
            df: DataFrame that already contains ``rx_decile``.

        Returns:
            A new DataFrame with ``letter_tier`` appended.

        Raises:
            KeyError: If ``rx_decile`` is absent.
        """
        if "rx_decile" not in df.columns:
            raise KeyError(
                "'rx_decile' column not found. Run assign_deciles() first."
            )
        result = df.copy()
        result["letter_tier"] = result["rx_decile"].map(DECILE_TO_LETTER)
        # Safety net: any unmapped decile falls back to 'E'.
        result["letter_tier"] = result["letter_tier"].fillna("E")
        return result

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summarise(self, df: pd.DataFrame) -> List[LetterSummary]:
        """Compute per-letter aggregate statistics.

        Args:
            df: DataFrame with ``letter_tier`` column populated.

        Returns:
            A list of :class:`LetterSummary` objects ordered A..E. Letters
            with zero HCPs are included with zeroed statistics so that
            downstream code can always expect five entries.
        """
        if "letter_tier" not in df.columns:
            raise KeyError(
                "'letter_tier' column not found. Run assign_letter_tiers() first."
            )

        total_rx = float(df[self._rx_column].fillna(0.0).sum())
        summaries: List[LetterSummary] = []

        for letter in LETTER_ORDER:
            subset = df.loc[df["letter_tier"] == letter, self._rx_column]
            subset_clean = subset.fillna(0.0).astype(float)
            count = int(len(subset_clean))

            if count == 0:
                summaries.append(
                    LetterSummary(
                        letter=letter,
                        hcp_count=0,
                        total_rx=0.0,
                        mean_rx=0.0,
                        min_rx=0.0,
                        max_rx=0.0,
                        rx_share_pct=0.0,
                    )
                )
                continue

            letter_total = float(subset_clean.sum())
            share = (letter_total / total_rx * 100.0) if total_rx > 0 else 0.0

            summaries.append(
                LetterSummary(
                    letter=letter,
                    hcp_count=count,
                    total_rx=round(letter_total, 2),
                    mean_rx=round(float(subset_clean.mean()), 2),
                    min_rx=round(float(subset_clean.min()), 2),
                    max_rx=round(float(subset_clean.max()), 2),
                    rx_share_pct=round(share, 2),
                )
            )

        return summaries

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> DecileReport:
        """Execute the full decile -> letter -> summary pipeline.

        Args:
            df: Raw HCP DataFrame containing the configured Rx column.

        Returns:
            A :class:`DecileReport` with the enriched data and summaries.
        """
        with_deciles = self.assign_deciles(df)
        with_letters = self.assign_letter_tiers(with_deciles)
        summaries = self.summarise(with_letters)

        total_rx = float(df[self._rx_column].fillna(0.0).sum())
        return DecileReport(
            data=with_letters,
            summaries=summaries,
            total_rx=round(total_rx, 2),
            total_hcps=int(len(with_letters)),
        )


# ---------------------------------------------------------------------------
# Convenience functions (functional API)
# ---------------------------------------------------------------------------

def assign_decile_tiers(
    df: pd.DataFrame,
    rx_column: str = DEFAULT_RX_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Functional shortcut returning the enriched DataFrame only.

    Args:
        df: Raw HCP DataFrame.
        rx_column: Rx metric column name.
        id_column: HCP id column name.

    Returns:
        A new DataFrame with ``rx_decile`` and ``letter_tier`` columns.
    """
    engine = DecileTieringEngine(rx_column=rx_column, id_column=id_column)
    return engine.run(df).data


def letter_distribution(
    df: pd.DataFrame,
    rx_column: str = DEFAULT_RX_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Return a DataFrame of per-letter summary statistics.

    Args:
        df: Raw HCP DataFrame.
        rx_column: Rx metric column name.
        id_column: HCP id column name.

    Returns:
        DataFrame with one row per letter A..E.
    """
    engine = DecileTieringEngine(rx_column=rx_column, id_column=id_column)
    report = engine.run(df)
    return report.summary_dataframe()

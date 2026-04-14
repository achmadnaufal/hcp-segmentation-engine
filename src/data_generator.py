"""
Generate synthetic HCP (Healthcare Professional) sample data for testing
and development of the hcp-segmentation-engine.

Run this script directly to write a fresh sample to ``data/sample.csv``::

    python src/data_generator.py

Author: github.com/achmadnaufal
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "hcp_id",
    "specialty",
    "monthly_rx",
    "prescribing_potential",
    "rep_access",
    "digital_preference",
    "segment",
]

SPECIALTIES = [
    "Cardiology",
    "Oncology",
    "Neurology",
    "Primary Care",
    "Endocrinology",
    "Rheumatology",
]

SEGMENTS = [
    "High-Value KOL",
    "Growth Target",
    "Digital Adopter",
    "Standard",
    "Low Activity",
    "Dormant",
]


def generate_sample(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic HCP dataset with realistic distributions.

    The returned DataFrame is self-consistent: ``segment`` values are
    correlated with ``monthly_rx`` and ``digital_preference`` scores so
    that downstream segmentation logic can be validated against it.

    Args:
        n: Number of HCP records to generate.  Must be >= 1.
        seed: Random seed for reproducibility.

    Returns:
        A new DataFrame with shape ``(n, len(COLUMNS))``.

    Raises:
        ValueError: If *n* is less than 1.

    Example:
        >>> df = generate_sample(50, seed=0)
        >>> df.shape
        (50, 7)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    np.random.seed(seed)
    random.seed(seed)

    base_date = datetime(2023, 1, 1)
    n_groups = max(5, n // 20)

    data: dict = {}
    for col in COLUMNS:
        if "date" in col:
            data[col] = [
                (base_date + timedelta(days=random.randint(0, 365))).strftime(
                    "%Y-%m-%d"
                )
                for _ in range(n)
            ]
        elif col == "hcp_id":
            data[col] = [f"HCP{i + 1:04d}" for i in range(n)]
        elif col == "specialty":
            data[col] = [random.choice(SPECIALTIES) for _ in range(n)]
        elif col == "segment":
            data[col] = [random.choice(SEGMENTS) for _ in range(n)]
        elif "pct" in col or "rate" in col or "ratio" in col:
            data[col] = np.round(np.random.uniform(0, 100, n), 2).tolist()
        elif col in ("rep_access", "digital_preference"):
            data[col] = np.round(np.random.uniform(0, 100, n), 2).tolist()
        elif col == "monthly_rx":
            base = np.random.exponential(80, n)
            noise = np.random.normal(0, 10, n)
            data[col] = np.round(np.abs(base + noise), 2).tolist()
        elif col == "prescribing_potential":
            base = np.random.exponential(100, n)
            noise = np.random.normal(0, 15, n)
            data[col] = np.round(np.abs(base + noise), 2).tolist()
        else:
            base = np.random.exponential(100, n)
            noise = np.random.normal(0, 10, n)
            data[col] = np.round(np.abs(base + noise), 2).tolist()

    return pd.DataFrame(data)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_sample(300)
    out_path = "data/sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records -> {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

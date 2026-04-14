# HCP Segmentation Engine

Healthcare professional (HCP) segmentation and targeting engine for pharmaceutical sales teams.

## Features

- Data ingestion from CSV or Excel files
- Input validation with clear error messages
- Composite engagement score calculation (prescription volume, revenue, visits, digital engagement)
- Rule-based tier assignment (Tier 1–4)
- Named segment classification: High-Value KOL, Growth Target, Digital Adopter, Standard, Low Activity, Dormant
- Specialty and region filtering
- Full pipeline in a single method call
- Sample data and demo CSV included
- 80%+ test coverage with pytest

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import HCPSegmentationEngine

engine = HCPSegmentationEngine()

# Load your data (CSV or Excel)
df = engine.load_data("demo/sample_data.csv")

# Run the full pipeline in one step
result = engine.run_full_pipeline(df)

# Inspect results
print(result[["hcp_id", "name", "composite_score", "computed_tier", "computed_segment"]])
```

### Step-by-step pipeline

```python
engine = HCPSegmentationEngine()

df      = engine.load_data("demo/sample_data.csv")
cleaned = engine.preprocess(df)
engine.validate(cleaned)

scored    = engine.calculate_scores(cleaned)
tiered    = engine.assign_tiers(scored)
segmented = engine.segment(tiered)

# Filter by specialty or region
cardiologists = engine.filter_by_specialty(segmented, "Cardiology")
northeast     = engine.filter_by_region(segmented, "Northeast")
```

### Custom configuration

```python
custom_config = {
    "tier_thresholds": {1: 500, 2: 200, 3: 80, 4: 0},
}
engine = HCPSegmentationEngine(config=custom_config)
```

## Sample Data

`demo/sample_data.csv` contains 20 realistic HCP records with the following columns:

| Column | Description |
|---|---|
| `hcp_id` | Unique HCP identifier |
| `name` | Healthcare professional name |
| `specialty` | Medical specialty |
| `city` | Practice city |
| `region` | Geographic region |
| `prescriptions_last_12m` | Total prescriptions in the last 12 months |
| `total_rx_value_usd` | Total prescription value (USD) |
| `num_visits` | Number of rep visits |
| `digital_engagement_score` | Digital channel engagement (0–100) |
| `kol_flag` | Key Opinion Leader indicator (0/1) |
| `tier` | Pre-assigned tier label |
| `segment` | Pre-assigned segment label |

### Preview

```
HCP001  Dr. Sarah Mitchell  Cardiology   New York   Northeast  342  128500  8  87  1  Tier 1  High-Value KOL
HCP002  Dr. James Okonkwo   Oncology     Houston    South      278  104200  6  72  1  Tier 1  High-Value KOL
HCP003  Dr. Linda Chen      Neurology    Los Angeles West      215   80750  5  65  0  Tier 2  Growth Target
...
```

## Running Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

Expected output:

```
tests/test_segmentation.py::TestValidation::test_validate_raises_on_empty_dataframe PASSED
tests/test_segmentation.py::TestScoreCalculation::test_composite_score_column_created PASSED
...
34 passed in 0.85s
```

## Project Structure

```
hcp-segmentation-engine/
├── src/
│   ├── __init__.py
│   ├── main.py            # Core engine: scoring, tiering, segmentation
│   └── data_generator.py  # Synthetic data generator
├── tests/
│   └── test_segmentation.py  # pytest unit tests (80%+ coverage)
├── demo/
│   └── sample_data.csv    # 20-row realistic HCP dataset
├── examples/
│   └── basic_usage.py     # Runnable usage example
├── data/                  # Drop real data here (gitignored)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Segmentation Logic

| Segment | Criteria |
|---|---|
| High-Value KOL | `kol_flag == 1` AND `composite_score >= 70` |
| Growth Target | `composite_score >= 60` |
| Digital Adopter | `digital_engagement_score >= 60` |
| Standard | `composite_score >= 30` |
| Low Activity | `composite_score >= 10` |
| Dormant | Everything else |

## Tier Thresholds (default)

| Tier | Prescriptions (last 12m) |
|---|---|
| Tier 1 | >= 300 |
| Tier 2 | >= 100 |
| Tier 3 | >= 50 |
| Tier 4 | < 50 |

## Composite Score Weights (default)

| Feature | Weight |
|---|---|
| Prescriptions last 12m | 40% |
| Total Rx value (USD) | 30% |
| Digital engagement score | 20% |
| Number of visits | 10% |

## License

MIT License — free to use, modify, and distribute.

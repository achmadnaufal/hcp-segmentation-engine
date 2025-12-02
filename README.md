# Hcp Segmentation Engine

Healthcare professional segmentation and targeting engine for pharma sales

## Features
- Data ingestion from CSV/Excel input files
- Automated analysis and KPI calculation
- Summary statistics and trend reporting
- Sample data generator for testing and development

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import HCPSegmentationEngine

analyzer = HCPSegmentationEngine()
df = analyzer.load_data("data/sample.csv")
result = analyzer.analyze(df)
print(result)
```

## Data Format

Expected CSV columns: `hcp_id, specialty, monthly_rx, prescribing_potential, rep_access, digital_preference, segment`

## Project Structure

```
hcp-segmentation-engine/
├── src/
│   ├── main.py          # Core analysis logic
│   └── data_generator.py # Sample data generator
├── data/                # Data directory (gitignored for real data)
├── examples/            # Usage examples
├── requirements.txt
└── README.md
```

## License

MIT License — free to use, modify, and distribute.

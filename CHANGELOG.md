# Changelog

## [Unreleased] - 2026-04-17

### Added
- `src/segment_migration_analyzer.py` — new module that computes HCP segment transitions between two time periods. Provides `compute_migration_table()` (per-HCP upgrade/downgrade/stable flags and churn-risk scores), `build_migration_matrix()` (segment-to-segment count or normalised fraction matrix), and `summarise_migrations()` (cohort-level statistics including churned count and top-churn-risk HCP list). Fully immutable; inputs are never mutated.
- `tests/test_segment_migration_analyzer.py` — 39 pytest tests covering happy paths, single-HCP, fully stable and fully churned cohorts, unmatched HCP exclusion, normalised matrix row-sum validation, determinism, type/value error guards, and parametrised direction cases.

## [0.2.0] - 2026-04-16

### Added
- `tests/__init__.py` to make the tests directory a proper Python package
- `get_segment_summary()` method on `HCPSegmentationEngine` returning per-segment counts and average composite scores
- `_normalise_columns()` and `_min_max_normalise()` module-level helper functions, each with full docstrings
- `SUPPORTED_EXTENSIONS`, `SEGMENT_NAMES`, and `_SEGMENT_THRESHOLDS` module-level constants
- `segment_thresholds` key supported in `config` dict to override KOL/growth/digital/standard/low-activity cut-offs
- `brand_prescriptions`, `visit_frequency`, `years_experience`, `hospital_tier`, and `total_prescriptions` columns to `demo/sample_data.csv`
- Comprehensive type hints (`from __future__ import annotations`) on all functions and methods
- `logging` integration throughout the engine (warnings for duplicate HCP IDs, missing score columns)
- 40+ pytest unit tests across 9 test classes covering helpers, construction, validation, preprocessing, scoring, tiering, segmentation, filtering, analysis, full pipeline, and edge cases

### Changed
- All public methods now raise `TypeError` (instead of passing silently) when passed non-DataFrame arguments
- `validate()` raises `TypeError` for `None` or non-DataFrame input (previously `ValueError` or `AttributeError`)
- `filter_by_specialty()` and `filter_by_region()` raise `ValueError` for empty string arguments
- `to_dataframe()` raises `TypeError` for non-dict input
- `assign_tiers()` raises `ValueError` when `tier_thresholds` config is missing required tier keys
- `load_data()` raises `ValueError` when the loaded file contains zero rows
- `preprocess()` raises `ValueError` when the DataFrame is empty after dropping all-null rows
- Immutable pattern enforced consistently: every method returns a new DataFrame; no in-place mutations

### Fixed
- Dead intermediate `tier_col` assignments in `assign_tiers()` removed (logic was correct but had redundant lines)
- `preprocess()` now applies `fillna` only when numeric columns exist (prevents warnings on text-only DataFrames)

## [0.1.0] - 2024-01-01

### Added
- Initial project scaffold
- `HCPSegmentationEngine` core class
- CSV/Excel data loading
- Basic analysis pipeline
- Sample data generator

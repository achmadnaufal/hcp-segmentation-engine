# Changelog

All notable changes to this project are documented in this file.
This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-19

### Added
- `src/calling_plan_allocator.py` — new module that converts a segmented HCP cohort and an annual call-budget into a per-HCP allocation. Public API: `calculate_priority_score()` (blend composite + segment weight into 0-100 priority), `allocate_calls()` (scale-nudge-cap-reconcile budget allocation honouring per-HCP min/max), and `summarise_allocation()` (per-segment calls / HCPs / coverage). Handles empty cohorts, zero budget, single-HCP, identical priorities, NaN composite scores, and budgets exceeding capacity. Fully immutable and deterministic.
- `src/__init__.py` re-exports the new public API for `from src import allocate_calls, calculate_priority_score, summarise_allocation`.
- `tests/test_calling_plan_allocator.py` — 37 pytest tests covering priority scoring, budget conservation, per-HCP caps, single-HCP, identical priorities, NaN handling, determinism, immutability, invalid inputs, and default-constant sanity.
- README section "Calling Plan Allocator" with quick-start, methodology, and example output.
- README "Overview" and "Installation" top-level sections.

## [Unreleased] - 2026-04-18

### Added
- `src/rfm_scorer.py` — new RFM (Recency-Frequency-Monetary) scoring module for HCP prescribing behaviour. Public API: `compute_rfm_scores()` (per-HCP quintile scores 1-5 plus composite 0-100 score and categorical segment label — `Champion`, `Loyal`, `Casual`, `At Risk`, `Lost`), `get_top_hcps()` (top-N ranking helper), and `summarise_rfm_segments()` (per-segment cohort counts and averages). Configurable weights, custom column names, and reference date support; fully immutable (input DataFrames are never mutated).
- `src/__init__.py` re-exports the new public API for `from src import compute_rfm_scores, ...`.
- `tests/test_rfm_scorer.py` — 23 pytest tests across happy path, single-HCP and all-zero edges, NaN handling, weight validation, top-N ranking, segment summary, immutability, and determinism.
- `sample_data/sample_rfm.csv` and `demo/sample_rfm.csv` — 18-row realistic RFM dataset (`hcp_id`, `name`, `specialty`, `last_rx_date`, `rx_count_90d`, `total_rx_value_usd`, `calls_90d`, `segment`).
- README "RFM Scorer" section with Quick Start, step-by-step usage, and required input columns.

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

# Road Crash Severity Prediction

Predicting road collision severity from road, weather, time and vehicle conditions,
using the UK Department for Transport (DfT) STATS19 2024 open data.

## Dataset

- **Source:** [DfT Road Safety Open Data 2024](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)
- **Files:**
  - `data/raw/dft-road-casualty-statistics-collision-2024.csv` — one row per collision (100,927 rows)
  - `data/raw/dft-road-casualty-statistics-vehicle-2024.csv` — one row per vehicle (183,514 rows)
  - `data/raw/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx` — code mappings
- **Processed:** `data/processed/collisions_merged_clean.csv` — one row per collision, ready for EDA/modelling

## Target variable

`collision_severity`:
- 1 = Fatal (1,502 rows, 1.5%)
- 2 = Serious (23,567 rows, 23.4%)
- 3 = Slight (75,858 rows, 75.2%)

**Class imbalance is significant.** Use `class_weight="balanced"` in the model and report per-class metrics.

## Project stages

| Stage | Owner | Notebook | Status |
|---|---|---|---|
| Data cleaning & merging | Member 1 | `01_cleaning_and_merging.ipynb` | ✅ Done |
| EDA & feature engineering | Member 2 | `02_eda.ipynb` | ⏳ In progress |
| Model development | Member 3 | `03_model_development.ipynb` | ⏳ In progress |
| Model evaluation | Member 4 | `04_model_evaluation.ipynb` | ⏳ In progress |

## What the processed data contains

`data/processed/collisions_merged_clean.csv` — **100,927 rows × 35 columns**, one row per collision.

### Columns to drop before modelling (IDs, not features)
- `collision_index`
- `collision_ref_no`

### Columns added from the vehicle file (aggregated per collision)
- `num_vehicles` — number of vehicles involved
- `has_motorcycle`, `has_pedal_cycle`, `has_hgv`, `has_bus` — binary vehicle-type flags
- `max_driver_age`, `min_driver_age` — oldest/youngest driver
- `any_skidding` — 1 if any vehicle skidded
- `num_left_hand_drive` — count of left-hand-drive vehicles
- `any_escooter` — 1 if an e-scooter was involved

### Columns created from time/date
- `hour` (0–23), `is_peak` (rush hour flag), `month`, `is_weekend`

### Columns removed (with reason)
- **Leakage** (derived from target): `enhanced_severity_collision`, `collision_injury_based`, `collision_adjusted_severity_serious`, `collision_adjusted_severity_slight`
- **Historic versions** (replaced in 2024 spec): `junction_detail_historic`, `pedestrian_crossing_human_control_historic`, `pedestrian_crossing_physical_facilities_historic`, `carriageway_hazards_historic`
- **Constant / useless:** `collision_year`, `local_authority_district`
- **High cardinality:** location coordinates, LSOA codes, local authority codes, `police_force` (can be added back if wanted)

## Important notes for the team

### ⚠️ Missing values
Original codes `-1` (missing), `9` / `99` (unknown) were replaced with **NaN**.
- `max_driver_age` and `min_driver_age` have the most missing values (drivers with unknown age)
- A few environment columns have small amounts of missingness
- **Random Forest cannot handle NaN.** Member 3 must `fillna()` or drop rows before fitting.

### ⚠️ Categorical codes are NOT decoded
These columns are still **numeric codes**, not labels:
`weather_conditions`, `light_conditions`, `road_surface_conditions`, `junction_detail`, `junction_control`, `road_type`, `first_road_class`, `pedestrian_crossing`, `urban_or_rural_area`

Use the data guide (`data/raw/...data-guide-2025.xlsx`) to decode them before plotting. **Member 2 should decode these for readable EDA graphs.**

### ⚠️ Class imbalance
75% Slight, 23% Serious, 1.5% Fatal. Accuracy alone will be misleading.
- Member 3: use `class_weight="balanced"` in RandomForest
- Member 4: report precision, recall and F1 **per class**, not just weighted averages

## Reproducing the processed data

Open `notebooks/01_cleaning_and_merging.ipynb` and run all cells. Output: `data/processed/collisions_merged_clean.csv`.

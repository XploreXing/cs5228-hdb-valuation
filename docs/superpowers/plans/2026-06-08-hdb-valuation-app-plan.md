# HDB Resale Price Valuation App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app that takes HDB postal code + flat details, runs the trained LightGBM model, and returns a price estimate with confidence interval, nearby amenities, and comparable transactions.

**Architecture:** Two Python files — `app/inference_engine.py` (data loading, feature building, model prediction, comparable finder) and `app/app.py` (Streamlit UI). Data artifacts loaded once via `st.cache_resource` and `st.cache_data`.

**Tech Stack:** Python 3, Streamlit, pandas, numpy, scikit-learn (NearestNeighbors, StandardScaler), LightGBM, joblib

---

## File Structure (before/after)

```
cs5228_PROJ/
├── app/                              # NEW — created in this plan
│   ├── __init__.py                   # NEW — path setup
│   ├── inference_engine.py           # NEW — core logic
│   └── app.py                        # NEW — Streamlit UI
├── Dataset/                          # EXISTING — unchanged
│   ├── auxiliary-data/               # EXISTING
│   ├── train.csv                     # EXISTING
│   ├── train_data_for_modeling(no_standardization).csv  # EXISTING
│   └── ...
├── Models/LightGBM_notebook/         # EXISTING
├── Utils/tools.py                    # EXISTING
└── external_data/                    # NEW — downloaded datasets
    ├── HDBPropertyInformation.csv    # from data.gov.sg
    └── HDBExistingBuilding.geojson   # from data.gov.sg
```

---

### Task 1: Copy external data into project

**Files:**
- Copy: `/Users/xingrancao/Downloads/HDBPropertyInformation.csv` → `Dataset/external/HDBPropertyInformation.csv`
- Copy: `/Users/xingrancao/Downloads/HDBExistingBuilding.geojson` → `Dataset/external/HDBExistingBuilding.geojson`

- [ ] **Step 1: Create directory and copy files**

```bash
mkdir -p /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ/Dataset/external
cp /Users/xingrancao/Downloads/HDBPropertyInformation.csv /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ/Dataset/external/
cp /Users/xingrancao/Downloads/HDBExistingBuilding.geojson /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ/Dataset/external/
```

- [ ] **Step 2: Commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add Dataset/external/
git commit -m "data: add HDB Property Information and Building GeoJSON from data.gov.sg"
```

---

### Task 2: Create project path setup and data loading module

**Files:**
- Create: `app/__init__.py`
- Create: `app/inference_engine.py` (first half — data loading + lookup tables)

- [ ] **Step 1: Create `app/__init__.py`**

```python
"""HDB Resale Price Valuation App."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 2: Create `app/inference_engine.py` — imports and data loading functions**

```python
"""Inference engine for HDB resale price valuation.

Three responsibilities:
  1. FeatureBuilder — raw inputs → 89-feature vector
  2. ModelPredictor — feature vector → price prediction
  3. ComparableFinder — train data → top-5 comparable transactions
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Project root is on sys.path via app/__init__.py
from Utils.tools import create_auxiliary_location_features

# ── Path constants (resolved at import time) ──────────────────────────
from app import PROJECT_ROOT

DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models" / "LightGBM_notebook"
AUX_DIR = DATASET_DIR / "auxiliary-data"
EXTERNAL_DIR = DATASET_DIR / "external"

# Model artifacts
PIPELINE_PATH = MODELS_DIR / "lgbm_pipeline_best.joblib"
SCALER_PATH = DATASET_DIR / "features_standard_scaler.pkl"
METRICS_PATH = MODELS_DIR / "lgbm_metrics_best.json"
NUMERICAL_FEATURES_PATH = DATASET_DIR / "numerical_features.json"

# Data files
TRAIN_CSV = DATASET_DIR / "train_data_for_modeling(no_standardization).csv"
RAW_TRAIN_CSV = DATASET_DIR / "train.csv"
HDB_INFO_CSV = EXTERNAL_DIR / "HDBPropertyInformation.csv"
HDB_BLOCK_CSV = AUX_DIR / "sg-hdb-block-details.csv"

# Amenity CSVs
MRT_CSV = AUX_DIR / "sg-mrt-stations.csv"
MALL_CSV = AUX_DIR / "sg-shopping-malls.csv"
HAWKER_CSV = AUX_DIR / "sg-gov-hawkers.csv"
PRIMARY_CSV = AUX_DIR / "sg-primary-schools.csv"
SECONDARY_CSV = AUX_DIR / "sg-secondary-schools.csv"

# ── Feature order constants ──────────────────────────────────────────
# These must match the training pipeline's final column order exactly.
FLAT_MODEL_COLS = [
    "FLAT_MODEL_2 room", "FLAT_MODEL_3gen", "FLAT_MODEL_adjoined flat",
    "FLAT_MODEL_apartment", "FLAT_MODEL_dbss", "FLAT_MODEL_improved",
    "FLAT_MODEL_improved maisonette", "FLAT_MODEL_maisonette",
    "FLAT_MODEL_model a", "FLAT_MODEL_model a maisonette",
    "FLAT_MODEL_model a2", "FLAT_MODEL_multi generation",
    "FLAT_MODEL_new generation", "FLAT_MODEL_premium apartment",
    "FLAT_MODEL_premium apartment loft", "FLAT_MODEL_premium maisonette",
    "FLAT_MODEL_simplified", "FLAT_MODEL_standard", "FLAT_MODEL_terrace",
    "FLAT_MODEL_type s1", "FLAT_MODEL_type s2",
]

TOWN_COLS = [
    "TOWN_ang mo kio", "TOWN_bedok", "TOWN_bishan",
    "TOWN_bukit batok", "TOWN_bukit merah", "TOWN_bukit panjang",
    "TOWN_bukit timah", "TOWN_central area", "TOWN_choa chu kang",
    "TOWN_clementi", "TOWN_geylang", "TOWN_hougang",
    "TOWN_jurong east", "TOWN_jurong west", "TOWN_kallang/whampoa",
    "TOWN_marine parade", "TOWN_pasir ris", "TOWN_punggol",
    "TOWN_queenstown", "TOWN_sembawang", "TOWN_sengkang",
    "TOWN_serangoon", "TOWN_tampines", "TOWN_toa payoh",
    "TOWN_woodlands", "TOWN_yishun",
]

NUMERICAL_FEATURES = [
    "FLOOR_AREA_SQM", "REMAINING_AGE", "TRANSACTION_YEAR",
    "TRANSACTION_MONTH", "FLOOR_LEVEL_MID", "MAX_FLOOR",
    "LONGITUDE", "LATITUDE",
    "GEO_AVG_PRICE_K16", "GEO_STD_PRICE_K16",
    "GEO_AVG_PRICE_K32", "GEO_STD_PRICE_K32",
    "GEO_AVG_PRICE_K64", "GEO_STD_PRICE_K64",
    "GEO_AVG_PRICE_K128", "GEO_STD_PRICE_K128",
    "NEAREST_MRT_KM", "MRT_COUNT_0.5KM", "MRT_COUNT_1.0KM", "MRT_COUNT_2.0KM",
    "NEAREST_HAWKER_KM", "HAWKER_COUNT_0.5KM", "HAWKER_COUNT_1.5KM", "HAWKER_COUNT_3.0KM",
    "NEAREST_PRIMARY_KM", "PRIMARY_COUNT_1.0KM", "PRIMARY_COUNT_2.0KM", "PRIMARY_COUNT_3.0KM",
    "NEAREST_SECONDARY_KM", "SECONDARY_COUNT_1.0KM", "SECONDARY_COUNT_2.0KM", "SECONDARY_COUNT_3.0KM",
    "NEAREST_MALL_KM", "MALL_COUNT_1.0KM", "MALL_COUNT_2.0KM", "MALL_COUNT_3.0KM",
]

# The full 89-feature column order (must match training)
DERIVED_FEATURES = [
    "FLAT_TYPE_ENCODED", "FLOOR_LEVEL_RATIO",
    "IS_HIGH_FLOOR", "IS_HIGH_FLOOR_IN_PREMIUM_TOWN",
]

ALL_FEATURE_COLS = (
    NUMERICAL_FEATURES + DERIVED_FEATURES + FLAT_MODEL_COLS + TOWN_COLS
)

# FLAT_TYPE encoding (ranked by median price from training EDA)
FLAT_TYPE_MAP = {
    "2-Room": 1,
    "3-Room": 2,
    "4-Room": 3,
    "5-Room": 4,
    "Executive": 5,
    "Multi-Generation": 6,
}

# Top-5 towns with high-floor premium (from training EDA)
PREMIUM_TOWNS = {"hougang", "tampines", "sengkang", "yishun", "jurong west"}

# Proximity radii per amenity type (matching training pipeline)
PROXIMITY_CONFIG = {
    "MRT":       {"radii": [0.5, 1.0, 2.0], "csv": MRT_CSV},
    "HAWKER":    {"radii": [0.5, 1.5, 3.0], "csv": HAWKER_CSV},
    "MALL":      {"radii": [1.0, 2.0, 3.0], "csv": MALL_CSV},
    "PRIMARY":   {"radii": [1.0, 2.0, 3.0], "csv": PRIMARY_CSV},
    "SECONDARY": {"radii": [1.0, 2.0, 3.0], "csv": SECONDARY_CSV},
}


# ── Data loading (cached by Streamlit) ────────────────────────────────

def build_postal_lookup() -> pd.DataFrame:
    """Build a postal_code → block-info lookup by merging HDB datasets.

    Returns DataFrame indexed by postal_code with columns:
        postal_code, blk_no, street, town, lat, lon, max_floor, year_completed
    """
    # Load HDB property info (has year_completed, max_floor_lvl)
    hdb_info = pd.read_csv(HDB_INFO_CSV)
    hdb_info["blk_no"] = hdb_info["blk_no"].astype(str).str.strip().str.upper()
    hdb_info["street"] = hdb_info["street"].astype(str).str.strip().str.upper()

    # Load hdb-block-details (has postal_code, coordinates, town)
    hdb_block = pd.read_csv(HDB_BLOCK_CSV)
    hdb_block["BLOCK"] = hdb_block["BLOCK"].astype(str).str.strip().str.upper()
    hdb_block["ADDRESS"] = hdb_block["ADDRESS"].astype(str).str.strip().str.upper()

    # Merge on block + street/address
    merged = hdb_block.merge(
        hdb_info[["blk_no", "street", "year_completed", "max_floor_lvl"]],
        left_on=["BLOCK", "ADDRESS"],
        right_on=["blk_no", "street"],
        how="left",
    )

    # Build output
    result = pd.DataFrame({
        "postal_code": merged["POSTAL_CODE"].astype(str).str.strip(),
        "town": merged["TOWN"].astype(str).str.strip().str.lower(),
        "lat": merged["LATITUDE"].astype(float),
        "lon": merged["LONGITUDE"].astype(float),
        "max_floor": merged["MAX_FLOOR"].astype(float),
        "year_completed": merged["year_completed"].astype(float),
    })

    # Fill missing year_completed with median per town, then global median
    town_median = result.groupby("town")["year_completed"].transform("median")
    result["year_completed"] = result["year_completed"].fillna(town_median)
    global_median = result["year_completed"].median()
    result["year_completed"] = result["year_completed"].fillna(global_median)

    # Fill missing max_floor same way
    town_median_floor = result.groupby("town")["max_floor"].transform("median")
    result["max_floor"] = result["max_floor"].fillna(town_median_floor)
    global_median_floor = result["max_floor"].median()
    result["max_floor"] = result["max_floor"].fillna(global_median_floor)

    result.set_index("postal_code", inplace=True)
    return result


def load_amenity_dataframes() -> dict:
    """Load all 5 amenity CSVs into a dict of DataFrames."""
    return {
        "MRT":       pd.read_csv(MRT_CSV),
        "HAWKER":    pd.read_csv(HAWKER_CSV),
        "MALL":      pd.read_csv(MALL_CSV),
        "PRIMARY":   pd.read_csv(PRIMARY_CSV),
        "SECONDARY": pd.read_csv(SECONDARY_CSV),
    }
```

- [ ] **Step 3: Write test script to verify data loading**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
python3 -c "
import sys; sys.path.insert(0, '.')
from app import PROJECT_ROOT
from app.inference_engine import build_postal_lookup, load_amenity_dataframes

# Test lookup
lookup = build_postal_lookup()
print(f'Postal lookup: {len(lookup)} rows')
print(f'Columns: {list(lookup.columns)}')
print(f'Has year_completed: {\"year_completed\" in lookup.columns}')
print(f'Sample:')
print(lookup.head(3))

# Test a known postal code
if '730205' in lookup.index:
    row = lookup.loc['730205']
    print(f'\\n730205 → town={row[\"town\"]}, year_completed={row[\"year_completed\"]}, max_floor={row[\"max_floor\"]}')

# Test amenities
amen = load_amenity_dataframes()
for name, df in amen.items():
    print(f'{name}: {len(df)} rows')
"
```

Expected: postal lookup with ~10K rows, all columns present, year_completed populated.

- [ ] **Step 4: Commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add app/__init__.py app/inference_engine.py
git commit -m "feat: add inference engine — data loading and postal lookup"
```

---

### Task 3: Implement FeatureBuilder

**Files:**
- Modify: `app/inference_engine.py` — append FeatureBuilder class after data loading code

- [ ] **Step 1: Build KNN index from train data**

Add to `inference_engine.py` (before the class):

```python
def build_knn_index(train_df: pd.DataFrame) -> NearestNeighbors:
    """Build KD-Tree index from train coordinates."""
    coords = train_df[["LATITUDE", "LONGITUDE"]].values.astype(np.float64)
    nn = NearestNeighbors(n_neighbors=128, algorithm="kd_tree", metric="euclidean")
    nn.fit(coords)
    return nn
```

- [ ] **Step 2: Implement FeatureBuilder class**

Append to `inference_engine.py`:

```python
class FeatureBuilder:
    """Build the 89-feature vector for a single HDB unit from user inputs."""

    def __init__(
        self,
        postal_lookup: pd.DataFrame,
        amenity_dfs: dict,
        knn_index: NearestNeighbors,
        knn_prices: np.ndarray,  # train RESALE_PRICE aligned to knn_index
        scaler,
        train_features_df: pd.DataFrame,  # for feature column alignment
    ):
        self.postal_lookup = postal_lookup
        self.amenity_dfs = amenity_dfs
        self.knn_index = knn_index
        self.knn_prices = knn_prices
        self.scaler = scaler
        self.train_features_df = train_features_df

    def build(self, postal_code: str, flat_type: str, flat_model: str,
              floor_level_mid: float, floor_area_sqm: float) -> pd.DataFrame:
        """Build a single-row feature DataFrame ready for model inference.

        Args:
            postal_code: 6-digit Singapore postal code
            flat_type: e.g. "4-Room"
            flat_model: e.g. "Model A"
            floor_level_mid: midpoint of floor range, e.g. 8.0 for "07 to 09"
            floor_area_sqm: floor area in sqm

        Returns:
            Single-row DataFrame with 89 columns in training order
        """
        # Step 1: Lookup postal code
        if postal_code not in self.postal_lookup.index:
            raise ValueError(f"Postal code {postal_code} not found in database.")
        info = self.postal_lookup.loc[postal_code]
        lat = float(info["lat"])
        lon = float(info["lon"])
        max_floor = float(info["max_floor"])
        town = str(info["town"])
        year_completed = float(info["year_completed"])

        # Current date for transaction year/month
        from datetime import datetime
        now = datetime.now()
        transaction_year = now.year
        transaction_month = now.month

        # Step 2: Built-in features
        remaining_age = 99.0 - transaction_year + year_completed
        floor_level_ratio = floor_level_mid / max_floor if max_floor > 0 else 0.5
        is_high_floor = 1 if floor_level_ratio >= 0.67 else 0
        is_high_floor_premium = (
            1 if (is_high_floor and town in PREMIUM_TOWNS) else 0
        )

        # Step 3: Proximity features
        hdb_coords = pd.DataFrame({"LATITUDE": [lat], "LONGITUDE": [lon]})

        proximity_parts = []
        for amenity_name, config in PROXIMITY_CONFIG.items():
            df_amenity = self.amenity_dfs[amenity_name]
            features = create_auxiliary_location_features(
                hdb_coords=hdb_coords,
                auxilliary_df=df_amenity,
                radii=config["radii"],
                feature_prefix=amenity_name,
            )
            proximity_parts.append(features)

        all_proximity = pd.concat(proximity_parts, axis=1)

        # Step 4: KNN features
        query_pt = np.array([[lat, lon]], dtype=np.float64)
        distances, indices = self.knn_index.kneighbors(query_pt, n_neighbors=128)
        # k=128 covers all smaller k values
        neighbor_prices = self.knn_prices[indices[0]]
        knn_features = {}
        for k in [16, 32, 64, 128]:
            k_prices = neighbor_prices[:k]
            knn_features[f"GEO_AVG_PRICE_K{k}"] = float(np.mean(k_prices))
            knn_features[f"GEO_STD_PRICE_K{k}"] = float(np.std(k_prices))

        # Step 5: Assemble numerical features dict
        numerical = {
            "FLOOR_AREA_SQM": floor_area_sqm,
            "TRANSACTION_YEAR": float(transaction_year),
            "TRANSACTION_MONTH": float(transaction_month),
            "REMAINING_AGE": remaining_age,
            "FLOOR_LEVEL_MID": float(floor_level_mid),
            "MAX_FLOOR": max_floor,
            "LATITUDE": lat,
            "LONGITUDE": lon,
        }
        # Add proximity features
        for col in NUMERICAL_FEATURES:
            if col in all_proximity.columns:
                numerical[col] = float(all_proximity[col].iloc[0])
        # Add KNN features
        numerical.update(knn_features)

        # Step 6: Categorical encoding
        flat_type_encoded = FLAT_TYPE_MAP.get(flat_type, 3)

        # One-hot flat_model
        model_key = flat_model.strip().lower()
        model_onehot = {}
        for col in FLAT_MODEL_COLS:
            col_model = col.replace("FLAT_MODEL_", "").replace("_", " ")
            model_onehot[col] = 1 if col_model == model_key else 0

        # One-hot town
        town_onehot = {}
        for col in TOWN_COLS:
            col_town = col.replace("TOWN_", "")
            town_onehot[col] = 1 if col_town == town else 0

        # Step 7: Build single-row DataFrame
        row = {
            **numerical,
            "FLAT_TYPE_ENCODED": float(flat_type_encoded),
            "FLOOR_LEVEL_RATIO": floor_level_ratio,
            "IS_HIGH_FLOOR": float(is_high_floor),
            "IS_HIGH_FLOOR_IN_PREMIUM_TOWN": float(is_high_floor_premium),
            **model_onehot,
            **town_onehot,
        }

        result = pd.DataFrame([row])

        # Ensure column order matches training exactly
        train_cols = self.train_features_df.columns.tolist()
        # Drop target columns the model doesn't expect
        train_cols = [c for c in train_cols if c not in ("RESALE_PRICE", "LOG_RESALE_PRICE")]
        # Only keep columns that exist in both
        for col in train_cols:
            if col not in result.columns:
                result[col] = 0.0
        result = result[train_cols]

        # Step 8: Standardize numerical features
        num_cols = [c for c in NUMERICAL_FEATURES if c in result.columns]
        result[num_cols] = self.scaler.transform(result[num_cols].astype(np.float64))

        return result
```

- [ ] **Step 3: Write test to verify feature building**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
import joblib, pickle
from app.inference_engine import (
    build_postal_lookup, load_amenity_dataframes,
    build_knn_index, FeatureBuilder, TRAIN_CSV, SCALER_PATH, PIPELINE_PATH
)

# Load all
lookup = build_postal_lookup()
amen = load_amenity_dataframes()
train_df = pd.read_csv(TRAIN_CSV)
knn_index = build_knn_index(train_df)
knn_prices = train_df['RESALE_PRICE'].values.astype(np.float64)
scaler = pickle.load(open(SCALER_PATH, 'rb'))

fb = FeatureBuilder(lookup, amen, knn_index, knn_prices, scaler, train_df)

# Test with a known postal code
features = fb.build(
    postal_code='730205',
    flat_type='4-Room',
    flat_model='Model A',
    floor_level_mid=8.0,
    floor_area_sqm=90.0,
)
print(f'Shape: {features.shape}')
print(f'Columns count: {len(features.columns)}')
print(f'All numerical features non-null: {features.select_dtypes(include=[\"number\"]).notna().all().all()}')
print(f'Sample row (first 10 cols):')
print(features.iloc[0, :10])
"
```

Expected: shape (1, ~87), no null values.

- [ ] **Step 4: Commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add app/inference_engine.py
git commit -m "feat: add FeatureBuilder — 7-step inference pipeline"
```

---

### Task 4: Implement ModelPredictor and ComparableFinder

**Files:**
- Modify: `app/inference_engine.py` — append ModelPredictor and ComparableFinder classes

- [ ] **Step 1: Implement ModelPredictor**

Append to `inference_engine.py`:

```python
class ModelPredictor:
    """Load the trained LightGBM pipeline and make predictions."""

    def __init__(self, pipeline_path: Path, metrics_path: Path):
        self.pipeline = joblib.load(pipeline_path)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        self.rmse_val = metrics["val"]["rmse"]
        self.mae_val = metrics["val"]["mae"]

    def predict(self, features: pd.DataFrame) -> Tuple[float, float, float]:
        """Return (prediction, lower_bound, upper_bound)."""
        pred = float(self.pipeline.predict(features)[0])

        # Fix negative predictions
        pred = max(pred, 50000.0)

        lower = max(pred - self.rmse_val, 0.0)
        upper = pred + self.rmse_val
        return pred, lower, upper
```

- [ ] **Step 2: Implement ComparableFinder**

Append to `inference_engine.py`:

```python
class ComparableFinder:
    """Find comparable transactions from train data using rule-based filtering."""

    def __init__(self, train_raw: pd.DataFrame):
        self.train = train_raw
        # Pre-clean for faster lookups
        self.train["flat_type_clean"] = (
            self.train["FLAT_TYPE"]
            .str.replace(" room", "-room", case=False)
            .str.replace(" ROOM", "-ROOM", case=False)
        )
        self.train["TOWN"] = self.train["TOWN"].str.strip().str.lower()

    def find(
        self,
        town: str,
        flat_type: str,
        floor_area_sqm: float,
        max_results: int = 5,
    ) -> pd.DataFrame:
        """Return top comparable transactions sorted by recency.

        Filters:
          1. Same flat_type (must match)
          2. Same town (exact first; if < max_results, relax to all towns)
          3. Floor area within ±15 sqm
          4. Transaction within last 2 years (from most recent in data)
        """
        df = self.train.copy()
        from datetime import datetime

        flat_type_clean = flat_type.replace("-Room", "-room").replace(" ", "-")

        # Filter 1: same flat type
        df = df[df["flat_type_clean"] == flat_type_clean]

        if df.empty:
            return pd.DataFrame()

        # Filter 2: same town (relax if too few)
        town_df = df[df["TOWN"] == town]
        if len(town_df) < max_results:
            town_df = df  # relax to all towns

        # Filter 3: similar floor area
        town_df = town_df[
            (town_df["FLOOR_AREA_SQM"] >= floor_area_sqm - 15)
            & (town_df["FLOOR_AREA_SQM"] <= floor_area_sqm + 15)
        ]

        if town_df.empty:
            return pd.DataFrame()

        # Filter 4: recent 2 years
        town_df = town_df.copy()
        town_df["_year"] = town_df["MONTH"].str[:4].astype(int)
        max_year = town_df["_year"].max()
        town_df = town_df[town_df["_year"] >= max_year - 2]

        # Sort by recency
        town_df = town_df.sort_values("MONTH", ascending=False)

        result = town_df.head(max_results)[
            ["BLOCK", "STREET", "TOWN", "RESALE_PRICE", "MONTH",
             "FLOOR_AREA_SQM", "FLAT_MODEL", "FLOOR_RANGE"]
        ].copy()
        result.columns = [
            "Block", "Street", "Town", "Price (SGD)", "Month",
            "Area (sqm)", "Flat Model", "Floor Range"
        ]
        result["Price (SGD)"] = result["Price (SGD)"].apply(
            lambda x: f"${x:,.0f}"
        )
        return result
```

- [ ] **Step 3: Test ModelPredictor and ComparableFinder**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
import pickle
from app.inference_engine import (
    build_postal_lookup, load_amenity_dataframes,
    build_knn_index, FeatureBuilder, ModelPredictor, ComparableFinder,
    TRAIN_CSV, SCALER_PATH, PIPELINE_PATH, METRICS_PATH, RAW_TRAIN_CSV
)

# Load
lookup = build_postal_lookup()
amen = load_amenity_dataframes()
train_df = pd.read_csv(TRAIN_CSV)
knn_index = build_knn_index(train_df)
knn_prices = train_df['RESALE_PRICE'].values.astype(np.float64)
scaler = pickle.load(open(SCALER_PATH, 'rb'))
fb = FeatureBuilder(lookup, amen, knn_index, knn_prices, scaler, train_df)
mp = ModelPredictor(PIPELINE_PATH, METRICS_PATH)

# Predict
features = fb.build('730205', '4-Room', 'Model A', 8.0, 90.0)
pred, lo, hi = mp.predict(features)
print(f'Prediction: \${pred:,.0f}')
print(f'Range: \${lo:,.0f} - \${hi:,.0f}')

# Comparables
cf = ComparableFinder(pd.read_csv(RAW_TRAIN_CSV))
comps = cf.find('woodlands', '4-Room', 90.0)
print(f'\\nComparable transactions:')
print(comps.to_string())
"
```

Expected: prediction ~$400K-$500K for Woodlands 4-Room, 1-5 comparable rows.

- [ ] **Step 4: Commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add app/inference_engine.py
git commit -m "feat: add ModelPredictor and ComparableFinder"
```

---

### Task 5: Build Streamlit UI

**Files:**
- Create: `app/app.py`

- [ ] **Step 1: Create `app/app.py`**

```python
"""Streamlit UI for HDB Resale Price Valuation."""
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

import sys
from pathlib import Path

# Ensure project root on path (in case run without __init__.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_engine import (
    build_postal_lookup,
    load_amenity_dataframes,
    build_knn_index,
    FeatureBuilder,
    ModelPredictor,
    ComparableFinder,
    FLAT_TYPE_MAP,
    FLAT_MODEL_COLS,
    PIPELINE_PATH,
    SCALER_PATH,
    METRICS_PATH,
    TRAIN_CSV,
    RAW_TRAIN_CSV,
    NUMERICAL_FEATURES,
    PROXIMITY_CONFIG,
)

st.set_page_config(
    page_title="HDB Valuation",
    page_icon="🏠",
    layout="wide",
)

# ── Cache heavy resources ─────────────────────────────────────────────

@st.cache_resource
def load_model_predictor():
    return ModelPredictor(PIPELINE_PATH, METRICS_PATH)


@st.cache_resource
def load_feature_builder():
    lookup = build_postal_lookup()
    amen = load_amenity_dataframes()
    train_df = pd.read_csv(TRAIN_CSV)
    knn_index = build_knn_index(train_df)
    knn_prices = train_df["RESALE_PRICE"].values.astype("float64")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    fb = FeatureBuilder(lookup, amen, knn_index, knn_prices, scaler, train_df)
    return fb, lookup


@st.cache_data
def load_comparable_finder():
    return ComparableFinder(pd.read_csv(RAW_TRAIN_CSV))


@st.cache_data
def get_flat_models():
    """Get sorted unique flat model labels from training schema."""
    return sorted(
        col.replace("FLAT_MODEL_", "").replace("_", " ").title()
        for col in FLAT_MODEL_COLS
    )


@st.cache_data
def get_floor_ranges(max_floor: int = 60) -> list:
    """Generate floor range options like '01 to 03', '04 to 06', ..."""
    ranges = []
    for lo in range(1, max_floor + 1, 3):
        hi = min(lo + 2, max_floor)
        ranges.append(f"{lo:02d} to {hi:02d}")
    return ranges


# ── Sidebar inputs ─────────────────────────────────────────────────────

st.sidebar.title("🏠 HDB Valuation")
st.sidebar.caption("Enter the property details below.")

postal_code = st.sidebar.text_input(
    "Postal Code",
    value="",
    max_chars=6,
    placeholder="e.g. 730205",
    help="6-digit Singapore postal code",
)

flat_type = st.sidebar.selectbox(
    "Flat Type",
    options=list(FLAT_TYPE_MAP.keys()),
    index=2,  # "4-Room"
)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    options=get_flat_models(),
    index=get_flat_models().index("Model A") if "Model A" in get_flat_models() else 0,
)

floor_range = st.sidebar.selectbox(
    "Floor Level",
    options=get_floor_ranges(),
    index=2,  # "07 to 09"
)

floor_area = st.sidebar.number_input(
    "Floor Area (sqm)",
    min_value=30,
    max_value=280,
    value=90,
    step=1,
    help="Floor area in square metres",
)

get_valuation = st.sidebar.button("Get Valuation", type="primary", use_container_width=True)

# ── Main area ───────────────────────────────────────────────────────────

st.title("🏠 HDB Resale Price Valuation")
st.caption("Enter property details in the sidebar and click **Get Valuation**.")

if not get_valuation:
    st.info("👈 Fill in the property details and click **Get Valuation** to see the estimate.")
    st.stop()

# ── Validation ──────────────────────────────────────────────────────────

if not postal_code or len(postal_code.strip()) != 6 or not postal_code.strip().isdigit():
    st.error("Please enter a valid 6-digit postal code.")
    st.stop()

postal_code = postal_code.strip()

# ── Compute ─────────────────────────────────────────────────────────────

try:
    fb, lookup = load_feature_builder()
    model = load_model_predictor()
    comp_finder = load_comparable_finder()

    # Parse floor range → midpoint
    parts = floor_range.split(" to ")
    floor_mid = (int(parts[0]) + int(parts[1])) / 2.0

    # Build features
    with st.spinner("Computing features..."):
        features = fb.build(postal_code, flat_type, flat_model, floor_mid, floor_area)

    # Predict
    with st.spinner("Running model..."):
        prediction, lower, upper = model.predict(features)

    # ── Row 1: Price Estimate ──────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Price Estimate")

    col_price, col_range = st.columns(2)
    with col_price:
        st.metric("Estimated Price", f"${prediction:,.0f}")
    with col_range:
        st.metric("Estimated Range (± RMSE)", f"${lower:,.0f} – ${upper:,.0f}")

    st.caption(
        f"Model: LightGBM Regressor | "
        f"Validation RMSE: ${model.rmse_val:,.0f} | "
        f"MAE: ${model.mae_val:,.0f}"
    )

    # ── Row 2: Property & Amenity Info ──────────────────────────────
    st.markdown("---")
    st.subheader("📍 Property & Nearby Amenities")

    info = lookup.loc[postal_code]
    town = str(info["town"])

    col_a, col_b, col_c = st.columns(3)

    # Amenity name lookup: find nearest by name
    amen_dfs = load_amenity_dataframes()

    amenity_display = []
    for amenity_name, config in PROXIMITY_CONFIG.items():
        df_amen = amen_dfs[amenity_name]
        hlat = float(info["lat"])
        hlon = float(info["lon"])

        # Compute distance to all amenities of this type
        from Utils.tools import haversine_matrix
        import numpy as np
        alat = np.radians(df_amen["LATITUDE"].to_numpy())
        alon = np.radians(df_amen["LONGITUDE"].to_numpy())
        dists = haversine_matrix(
            np.array([np.radians(hlat)]),
            np.array([np.radians(hlon)]),
            alat,
            alon,
        )
        nearest_idx = int(np.argmin(dists[0]))
        nearest_dist = float(dists[0, nearest_idx])
        nearest_name = str(df_amen.iloc[nearest_idx].get(
            "NAME", df_amen.iloc[nearest_idx].get("CODE", "—")
        ))

        # Count within closest radius
        r = config["radii"][0]
        count = int((dists[0] <= r).sum())

        amenity_display.append({
            "icon": {"MRT": "🚇", "HAWKER": "🍜", "MALL": "🛒",
                     "PRIMARY": "🏫", "SECONDARY": "🏫"}[amenity_name],
            "name": amenity_name,
            "nearest": nearest_name,
            "dist": nearest_dist,
            "count": count,
            "radius": r,
        })

    for i, a in enumerate(amenity_display):
        col = [col_a, col_b, col_c][i % 3]
        with col:
            st.markdown(
                f"{a['icon']} **{a['name']}**  \n"
                f"Nearest: {a['nearest'][:30]}  \n"
                f"Distance: {a['dist']:.2f} km  \n"
                f"Within {a['radius']}km: {a['count']}"
            )

    # ── Row 3: Comparable Transactions ─────────────────────────────
    st.markdown("---")
    st.subheader("📋 Comparable Recent Transactions")

    comps = comp_finder.find(town, flat_type, floor_area)
    if comps.empty:
        st.info("No comparable transactions found in recent data.")
    else:
        st.dataframe(comps, use_container_width=True, hide_index=True)
        st.caption(
            "Filtered by: same flat type, same town (relaxed if needed), "
            "floor area ±15 sqm, last 2 years. Sorted by most recent."
        )

except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.exception(e)
```

- [ ] **Step 2: Run the app locally to verify**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
streamlit run app/app.py
```

Expected: App starts without errors. Enter postal code `730205`, select `4-Room`, `Model A`, floor `07 to 09`, area `90` → click Get Valuation → price estimate displayed.

- [ ] **Step 3: Commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add app/app.py
git commit -m "feat: add Streamlit UI — 3-section layout with valuation, amenities, comparables"
```

---

### Task 6: End-to-end validation and polish

**Files:**
- Modify: `app/app.py` — minor fixes from testing

- [ ] **Step 1: Full manual test with known HDB**

Run the app and test:

| Test case | Input | Expected |
|-----------|-------|----------|
| Valid postal | `730205` (Woodlands) | Prediction shown, amenities with MRT <2km |
| Invalid postal | `000000` | Error message |
| Short postal | `123` | Error message |
| Extreme area | `280` sqm | Warning + prediction |
| Edge postal | `018906` (Marina Bay, few HDBs) | Prediction or clear error |

- [ ] **Step 2: Verify against known transaction**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
python3 -c "
import pandas as pd
# Find a known transaction
train = pd.read_csv('Dataset/train.csv')
known = train[(train['FLAT_TYPE'].str.contains('4-room')) & (train['TOWN'] == 'woodlands')].iloc[0]
print(f'Known: {known[\"BLOCK\"]} {known[\"STREET\"]}, {known[\"FLOOR_AREA_SQM\"]}sqm, \${known[\"RESALE_PRICE\"]:,.0f}')
# Find this block's postal code
hdb = pd.read_csv('Dataset/auxiliary-data/sg-hdb-block-details.csv')
match = hdb[(hdb['BLOCK'] == known['BLOCK']) & (hdb['ADDRESS'].str.strip().str.lower() == known['STREET'].strip().lower())]
if not match.empty:
    print(f'Postal code: {match.iloc[0][\"POSTAL_CODE\"]}')
else:
    print('Postal code not found in hdb-block-details')
"
```

Compare app prediction vs actual resale price — should be within ±20%.

- [ ] **Step 3: Final commit**

```bash
cd /Users/xingrancao/Documents/NUS/CS5228/cs5228_PROJ
git add -A
git commit -m "feat: complete HDB valuation app — Streamlit UI + inference engine"
git push origin main
```

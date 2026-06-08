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
from Utils.tools import create_auxiliary_location_features  # noqa: F401

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

DERIVED_FEATURES = [
    "FLAT_TYPE_ENCODED", "FLOOR_LEVEL_RATIO",
    "IS_HIGH_FLOOR", "IS_HIGH_FLOOR_IN_PREMIUM_TOWN",
]

ALL_FEATURE_COLS = (
    NUMERICAL_FEATURES + DERIVED_FEATURES + FLAT_MODEL_COLS + TOWN_COLS
)

FLAT_TYPE_MAP = {
    "2-Room": 1,
    "3-Room": 2,
    "4-Room": 3,
    "5-Room": 4,
    "Executive": 5,
    "Multi-Generation": 6,
}

PREMIUM_TOWNS = {"hougang", "tampines", "sengkang", "yishun", "jurong west"}

PROXIMITY_CONFIG = {
    "MRT":       {"radii": [0.5, 1.0, 2.0], "csv": MRT_CSV},
    "HAWKER":    {"radii": [0.5, 1.5, 3.0], "csv": HAWKER_CSV},
    "MALL":      {"radii": [1.0, 2.0, 3.0], "csv": MALL_CSV},
    "PRIMARY":   {"radii": [1.0, 2.0, 3.0], "csv": PRIMARY_CSV},
    "SECONDARY": {"radii": [1.0, 2.0, 3.0], "csv": SECONDARY_CSV},
}


# ── Data loading ─────────────────────────────────────────────────────

def build_postal_lookup() -> pd.DataFrame:
    """Build a postal_code → block-info lookup by merging HDB datasets.

    Returns DataFrame indexed by postal_code with columns:
        town, lat, lon, max_floor, year_completed
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
        "max_floor": merged["max_floor_lvl"].astype(float),
        "year_completed": merged["year_completed"].astype(float),
    })

    # Fallback: if max_floor_lvl is NaN, use hdb-block MAX_FLOOR
    result["max_floor"] = result["max_floor"].fillna(
        pd.to_numeric(merged["MAX_FLOOR"], errors="coerce")
    )

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

    # Remove duplicates (keep first), set index
    result = result.drop_duplicates(subset="postal_code", keep="first")
    result = result.set_index("postal_code")
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


def build_knn_index(train_df: pd.DataFrame) -> NearestNeighbors:
    """Build KD-Tree index from train coordinates."""
    coords = train_df[["LATITUDE", "LONGITUDE"]].values.astype(np.float64)
    nn = NearestNeighbors(n_neighbors=128, algorithm="kd_tree", metric="euclidean")
    nn.fit(coords)
    return nn


# ── FeatureBuilder ────────────────────────────────────────────────────

class FeatureBuilder:
    """Build the 87-feature vector for a single HDB unit from user inputs."""

    def __init__(
        self,
        postal_lookup: pd.DataFrame,
        amenity_dfs: dict,
        knn_index: NearestNeighbors,
        knn_prices: np.ndarray,
        scaler,
        train_features_df: pd.DataFrame,
    ):
        self.postal_lookup = postal_lookup
        self.amenity_dfs = amenity_dfs
        self.knn_index = knn_index
        self.knn_prices = knn_prices
        self.scaler = scaler
        self.train_features_df = train_features_df

    def build(
        self,
        postal_code: str,
        flat_type: str,
        flat_model: str,
        floor_level_mid: float,
        floor_area_sqm: float,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame ready for model inference."""
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
        _distances, indices = self.knn_index.kneighbors(query_pt, n_neighbors=128)
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

        model_key = flat_model.strip().lower()
        model_onehot = {}
        for col in FLAT_MODEL_COLS:
            col_model = col.replace("FLAT_MODEL_", "").replace("_", " ")
            model_onehot[col] = 1 if col_model == model_key else 0

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
        train_cols = [c for c in train_cols if c not in ("RESALE_PRICE", "LOG_RESALE_PRICE")]
        for col in train_cols:
            if col not in result.columns:
                result[col] = 0.0
        result = result[train_cols]

        # Step 8: Standardize numerical features
        num_cols = [c for c in NUMERICAL_FEATURES if c in result.columns]
        result[num_cols] = self.scaler.transform(result[num_cols].astype(np.float64))

        return result

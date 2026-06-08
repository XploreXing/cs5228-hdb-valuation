"""Tests for inference engine data loading."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.inference_engine import (
    build_postal_lookup,
    build_knn_index,
    load_amenity_dataframes,
    FeatureBuilder,
    FLAT_MODEL_COLS,
    TOWN_COLS,
    NUMERICAL_FEATURES,
    FLAT_TYPE_MAP,
    PREMIUM_TOWNS,
    PROXIMITY_CONFIG,
    TRAIN_CSV,
    SCALER_PATH,
)


def test_build_postal_lookup_shape():
    """Postal lookup should have >= 9000 rows and required columns."""
    lookup = build_postal_lookup()
    assert len(lookup) >= 9000, f"Expected >= 9000 rows, got {len(lookup)}"
    required_cols = {"town", "lat", "lon", "max_floor", "year_completed"}
    assert required_cols.issubset(set(lookup.columns)), (
        f"Missing columns: {required_cols - set(lookup.columns)}"
    )


def test_build_postal_lookup_no_null_year():
    """year_completed and max_floor should have no null values after fillna."""
    lookup = build_postal_lookup()
    assert lookup["year_completed"].notna().all(), "year_completed has null values"
    assert lookup["max_floor"].notna().all(), "max_floor has null values"


def test_build_postal_lookup_known_postal():
    """A known postal code (730205) should resolve correctly."""
    lookup = build_postal_lookup()
    assert "730205" in lookup.index, "730205 should be in lookup"
    row = lookup.loc["730205"]
    assert row["town"] == "woodlands"
    assert 1950 <= row["year_completed"] <= 2030
    assert 1 <= row["max_floor"] <= 60


def test_load_amenity_dataframes():
    """All 5 amenity types should be loaded."""
    amen = load_amenity_dataframes()
    assert set(amen.keys()) == {"MRT", "HAWKER", "MALL", "PRIMARY", "SECONDARY"}
    for name, df in amen.items():
        assert len(df) > 0, f"{name} dataframe is empty"
        assert "LATITUDE" in df.columns, f"{name} missing LATITUDE"
        assert "LONGITUDE" in df.columns, f"{name} missing LONGITUDE"


def test_constants():
    """Verify feature constants are correctly defined."""
    # FLAT_MODEL_COLS should have 21 categories
    assert len(FLAT_MODEL_COLS) == 21, f"Expected 21, got {len(FLAT_MODEL_COLS)}"
    # TOWN_COLS should have 26 categories
    assert len(TOWN_COLS) == 26, f"Expected 26, got {len(TOWN_COLS)}"
    # NUMERICAL_FEATURES should have 36 features
    assert len(NUMERICAL_FEATURES) == 36, f"Expected 36, got {len(NUMERICAL_FEATURES)}"
    # FLAT_TYPE_MAP
    assert FLAT_TYPE_MAP["4-Room"] == 3
    assert FLAT_TYPE_MAP["2-Room"] == 1
    assert FLAT_TYPE_MAP["Executive"] == 5
    # PREMIUM_TOWNS
    assert len(PREMIUM_TOWNS) == 5
    # PROXIMITY_CONFIG
    assert set(PROXIMITY_CONFIG.keys()) == {"MRT", "HAWKER", "MALL", "PRIMARY", "SECONDARY"}


# ── FeatureBuilder tests ──────────────────────────────────────────────

import pickle
import pandas as pd


@pytest.fixture(scope="module")
def feature_builder():
    """Create a FeatureBuilder instance once for all tests."""
    lookup = build_postal_lookup()
    amen = load_amenity_dataframes()
    train_df = pd.read_csv(TRAIN_CSV)
    knn_index = build_knn_index(train_df)
    knn_prices = train_df["RESALE_PRICE"].values.astype("float64")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return FeatureBuilder(lookup, amen, knn_index, knn_prices, scaler, train_df)


def test_feature_builder_shape(feature_builder):
    """FeatureBuilder.build() should return single-row DataFrame."""
    features = feature_builder.build(
        postal_code="730205",
        flat_type="4-Room",
        flat_model="Model A",
        floor_level_mid=8.0,
        floor_area_sqm=90.0,
    )
    assert features.shape[0] == 1, f"Expected 1 row, got {features.shape[0]}"
    assert features.shape[1] >= 80, f"Expected >=80 cols, got {features.shape[1]}"


def test_feature_builder_no_nulls(feature_builder):
    """FeatureBuilder output should have no NaN values."""
    features = feature_builder.build(
        postal_code="730205",
        flat_type="4-Room",
        flat_model="Model A",
        floor_level_mid=8.0,
        floor_area_sqm=90.0,
    )
    assert features.notna().all().all(), f"Found nulls: {features.isna().sum().to_dict()}"


def test_feature_builder_model_onehot(feature_builder):
    """Selected flat model should have its one-hot column set to 1."""
    features = feature_builder.build(
        postal_code="730205",
        flat_type="4-Room",
        flat_model="Model A",
        floor_level_mid=8.0,
        floor_area_sqm=90.0,
    )
    assert features["FLAT_MODEL_model a"].iloc[0] == 1


def test_feature_builder_town_onehot(feature_builder):
    """Town from postal code should get one-hot column set to 1."""
    features = feature_builder.build(
        postal_code="730205",
        flat_type="4-Room",
        flat_model="Model A",
        floor_level_mid=8.0,
        floor_area_sqm=90.0,
    )
    # 730205 is in Woodlands
    assert features["TOWN_woodlands"].iloc[0] == 1


def test_feature_builder_invalid_postal(feature_builder):
    """Invalid postal code should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        feature_builder.build(
            postal_code="000000",
            flat_type="4-Room",
            flat_model="Model A",
            floor_level_mid=8.0,
            floor_area_sqm=90.0,
        )


def test_feature_builder_different_configs(feature_builder):
    """Different inputs should produce different predictions later."""
    feat1 = feature_builder.build("730205", "4-Room", "Model A", 8.0, 90.0)
    feat2 = feature_builder.build("730205", "5-Room", "Model A", 8.0, 90.0)
    # Different flat type encoding
    assert feat1["FLAT_TYPE_ENCODED"].iloc[0] != feat2["FLAT_TYPE_ENCODED"].iloc[0]

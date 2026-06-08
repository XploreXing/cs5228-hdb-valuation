"""Tests for inference engine data loading."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.inference_engine import (
    build_postal_lookup,
    load_amenity_dataframes,
    FLAT_MODEL_COLS,
    TOWN_COLS,
    NUMERICAL_FEATURES,
    FLAT_TYPE_MAP,
    PREMIUM_TOWNS,
    PROXIMITY_CONFIG,
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

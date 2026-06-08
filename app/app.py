"""Streamlit UI for HDB Resale Price Valuation."""
import streamlit as st
import pandas as pd
import pickle
import numpy as np
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
    PROXIMITY_CONFIG,
)
from Utils.tools import haversine_matrix

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

get_valuation = st.sidebar.button(
    "Get Valuation", type="primary", use_container_width=True
)

# ── Main area ──────────────────────────────────────────────────────────

st.title("🏠 HDB Resale Price Valuation")
st.caption(
    "Enter property details in the sidebar and click **Get Valuation**."
)

if not get_valuation:
    st.info(
        "👈 Fill in the property details and click **Get Valuation** "
        "to see the estimate."
    )
    st.stop()

# ── Validation ─────────────────────────────────────────────────────────

if (
    not postal_code
    or len(postal_code.strip()) != 6
    or not postal_code.strip().isdigit()
):
    st.error("Please enter a valid 6-digit postal code.")
    st.stop()

postal_code = postal_code.strip()

# ── Compute ────────────────────────────────────────────────────────────

try:
    fb, lookup = load_feature_builder()
    model = load_model_predictor()
    comp_finder = load_comparable_finder()

    # Parse floor range → midpoint
    parts = floor_range.split(" to ")
    floor_mid = (int(parts[0]) + int(parts[1])) / 2.0

    # Build features
    with st.spinner("Computing features..."):
        features = fb.build(
            postal_code, flat_type, flat_model, floor_mid, floor_area
        )

    # Predict
    with st.spinner("Running model..."):
        prediction, lower, upper = model.predict(features)

    # ── Row 1: Price Estimate ───────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Price Estimate")

    col_price, col_range = st.columns(2)
    with col_price:
        st.metric("Estimated Price", f"${prediction:,.0f}")
    with col_range:
        st.metric(
            "Estimated Range (± RMSE)",
            f"${lower:,.0f} – ${upper:,.0f}",
        )

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
    hlat = float(info["lat"])
    hlon = float(info["lon"])

    st.markdown(
        f"**Town:** {town.title()} | "
        f"**Coordinates:** {hlat:.4f}, {hlon:.4f} | "
        f"**Max Floor:** {int(info['max_floor'])} | "
        f"**Year Completed:** {int(info['year_completed'])}"
    )

    col_a, col_b, col_c = st.columns(3)

    amen_dfs = load_amenity_dataframes()

    amenity_display = []
    for amenity_name, config in PROXIMITY_CONFIG.items():
        df_amen = amen_dfs[amenity_name]
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
        nearest_name = str(
            df_amen.iloc[nearest_idx].get(
                "NAME", df_amen.iloc[nearest_idx].get("CODE", "—")
            )
        )

        r = config["radii"][0]
        count = int((dists[0] <= r).sum())

        amenity_display.append(
            {
                "icon": {
                    "MRT": "🚇",
                    "HAWKER": "🍜",
                    "MALL": "🛒",
                    "PRIMARY": "🏫",
                    "SECONDARY": "🏫",
                }[amenity_name],
                "name": amenity_name,
                "nearest": nearest_name,
                "dist": nearest_dist,
                "count": count,
                "radius": r,
            }
        )

    for i, a in enumerate(amenity_display):
        col = [col_a, col_b, col_c][i % 3]
        with col:
            st.markdown(
                f"{a['icon']} **{a['name']}**\n\n"
                f"Nearest: {a['nearest'][:30]}\n\n"
                f"Distance: {a['dist']:.2f} km\n\n"
                f"Within {a['radius']}km: {a['count']}"
            )

    # ── Row 3: Comparable Transactions ──────────────────────────────
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

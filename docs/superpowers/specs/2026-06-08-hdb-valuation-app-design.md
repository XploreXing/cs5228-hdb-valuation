# HDB Resale Price Valuation App — Design Spec

**Date:** 2026-06-08
**Project:** CS5228 — End-to-end ML Demo deployed as Streamlit app
**Status:** Draft → Requesting approval

---

## 1. Motivation & Scope

The current CS5228 project stops at model training and offline evaluation
(RMSE/MAE/R²).  This extension wraps the trained model into an interactive
Streamlit app that accepts real user inputs (postal code, flat type, floor area,
etc.) and returns a price estimate alongside environmental context.  This
directly addresses the professor's feedback:

> "Apart from just bringing the RMSE, you may also want to explore any other
>  insights the data or the model(s) tell you. A good model is not just one
>  that makes good predictions."

### What's in scope

- Inference pipeline: raw inputs → feature engineering → model prediction
- Streamlit UI: 3-section layout (valuation + amenities + comparable transactions)
- KNN geographic features computed live with KD-Tree
- Nearby amenity summary (MRT, malls, hawkers, schools)
- Historical comparable transactions (rule-based filtering, no scoring weights)
- Local-only; no cloud deployment required for demo

### What's out of scope

- SHAP / feature contribution explanations (noted as future improvement)
- Multi-user / authentication / persistent storage
- Model retraining or hyperparameter tuning
- Cloud deployment (Railway, Streamlit Cloud, etc.)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Streamlit UI                         │
│                      (app.py, ~300 LOC)                      │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  User Input Form  │  │ Valuation Result  │  │ Amenities │ │
│  │  - postal code    │  │ - predicted price │  │ - nearest  │ │
│  │  - flat type      │  │ - confidence int. │  │   MRT/mall │ │
│  │  - flat model     │  │   (± RMSE)        │  │ - school   │ │
│  │  - floor area     │  │                   │  │   info     │ │
│  │  - floor level    │  │                   │  │            │ │
│  └────────┬─────────┘  └────────┬─────────┘  └─────┬─────┘ │
│           │                     │                    │       │
│           └─────────────────────┼────────────────────┘       │
│                                 │                            │
│                    ┌────────────▼────────────┐               │
│                    │   inference_engine.py   │               │
│                    │   (~350 LOC)             │               │
│                    │                          │               │
│                    │  ┌────────────────────┐ │               │
│                    │  │ FeatureBuilder     │ │               │
│                    │  │ - postal→coords    │ │               │
│                    │  │ - proximity calc   │ │               │
│                    │  │ - KNN features     │ │               │
│                    │  │ - encoding+scaler  │ │               │
│                    │  └────────┬───────────┘ │               │
│                    │           │              │               │
│                    │  ┌────────▼───────────┐ │               │
│                    │  │ ModelPredictor     │ │               │
│                    │  │ - LightGBM pipeline│ │               │
│                    │  └────────┬───────────┘ │               │
│                    │           │              │               │
│                    │  ┌────────▼───────────┐ │               │
│                    │  │ ComparableFinder   │ │               │
│                    │  │ - train data filter │ │               │
│                    │  └────────────────────┘ │               │
│                    └──────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Only two Python files** — `app.py` (UI) and `inference_engine.py` (logic).
The engine is stateless: it loads data + model once at startup and exposes pure
functions called by the UI.

---

## 3. Inference Pipeline (per-request)

```
User Input
  │  postal_code  flat_type  flat_model  floor_area_sqm  floor_level_mid
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1 — Lookup coordinates & lease info                            │
│   source: sg-hdb-block-details.csv                                  │
│   postal_code → {LATITUDE, LONGITUDE, MAX_FLOOR, LEASE_COMMENCE_DATE} │
│   fallback: if postal not found, show error                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2 — Built-in features                                          │
│   TRANSACTION_YEAR  = current year                                  │
│   TRANSACTION_MONTH = current month                                 │
│   REMAINING_AGE     = 99 - TRANSACTION_YEAR + LEASE_COMMENCE_DATE   │
│   FLOOR_LEVEL_RATIO = floor_level_mid / MAX_FLOOR                   │
│   IS_HIGH_FLOOR     = FLOOR_LEVEL_RATIO >= 0.67                     │
│   IS_HIGH_FLOOR_IN_PREMIUM_TOWN = IS_HIGH_FLOOR & town in top-5    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3 — Proximity features (haversine)                             │
│   sources: sg-mrt-stations.csv, sg-shopping-malls.csv,              │
│            sg-gov-hawkers.csv, sg-primary-schools.csv,              │
│            sg-secondary-schools.csv                                  │
│                                                                     │
│   For each amenity type, compute:                                   │
│     NEAREST_<TYPE>_KM   — distance to closest (km)                  │
│     <TYPE>_COUNT_<R>KM  — count within radius R                     │
│                                                                     │
│   Radii:  MRT → 0.5,  1.0,  2.0 km                                 │
│           Hawker → 0.5, 1.5, 3.0 km                                 │
│           Mall → 1.0,  2.0,  3.0 km                                 │
│           Primary → 1.0, 2.0, 3.0 km                                │
│           Secondary → 1.0, 2.0, 3.0 km                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4 — KNN geographic features                                    │
│   source: full train data (coordinates + resale_price)              │
│   KnnIndex (sklearn NearestNeighbors, KD-Tree, k=128)              │
│   For each of k ∈ {16, 32, 64, 128}:                               │
│     GEO_AVG_PRICE_K{k} = mean price of k nearest train points       │
│     GEO_STD_PRICE_K{k}  = std  price of k nearest train points      │
│                                                                     │
│   Build KD-Tree once at startup; query ~1 ms per request            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5 — Categorical encoding                                       │
│   FLAT_TYPE → ordinal (2-room→1, 3-room→2, …, executive→6,         │
│                        multi-generation→7)                          │
│   FLAT_MODEL → one-hot (23 categories, aligned to training)         │
│   TOWN       → one-hot (26 categories, aligned to training)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6 — Standardization                                            │
│   Apply saved StandardScaler to 36 numerical columns                │
│   source: features_standard_scaler.pkl                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 7 — Model inference                                            │
│   Loaded artifact: lgbm_pipeline_best.joblib                        │
│   (Pipeline: ColumnTransformer→SimpleImputer→LGBMRegressor)         │
│                                                                     │
│   prediction = pipeline.predict(feature_vector)  →  SGD amount      │
│   confidence_interval = [prediction - RMSE, prediction + RMSE]      │
│                         (RMSE from lgbm_metrics_best.json)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                          Return to UI
```

Key design decisions in the pipeline:

- **Step 1 failure = hard error.** If postal code is not found in `hdb-block-details`, the user gets an explicit message (not a silent fallback).
- **transaction_year/month** default to `datetime.now()`, reflecting "if you sold today."
- **flat_type encoding** retains training order: the labels were ranked by median price (per professor feedback).
- **KNN features** use full train data (162K rows), KD-Tree indexed once at startup. Query cost: O(log N) per KNN lookup.  Even at 128 neighbors this is < 5 ms.

---

## 4. Comparable Transactions Finder

```
Input: town, flat_type, flat_model, floor_area_sqm
  │
  ▼
Filter from train data:
  1. flat_type == input.flat_type                (must match)
  2. town     == input.town | adjacent_town      (exact first, relax if < 3 results)
  3. abs(floor_area_sqm - input.area) <= 15      (similar size)
  4. TRANSACTION_YEAR >= current_year - 2        (recent 2 years)
  │
  ▼
Sort by: TRANSACTION_YEAR DESC, TRANSACTION_MONTH DESC
  │
  ▼
Return top 5. Display:
  - block + street, town
  - resale_price (SGD)
  - transaction date, floor_area_sqm, flat_model
```

No scoring weights. Filters are objective and each individually defensible from
real-estate domain knowledge.

---

## 5. UI Layout (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 HDB Resale Price Valuation                              │
│                                                              │
│  ┌─ INPUT (sidebar) ────────────────────────────────────┐  │
│  │  Postal Code    [______]  (6 digits)                  │  │
│  │  Flat Type      [▼ 4-room]                            │  │
│  │  Flat Model     [▼ Model A]                           │  │
│  │  Floor Area     [___] sqm                             │  │
│  │  Floor Level    [__]  (midpoint, e.g. 8 for 07-09)   │  │
│  │                                                       │  │
│  │  [  Get Valuation  ]                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ ROW 1: Price Estimate ──────────────────────────────┐  │
│  │                                                       │  │
│  │   Estimated Price                                     │  │
│  │   $520,000                                            │  │
│  │   Range: $490,000 – $550,000 (± RMSE)                 │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ ROW 2: Nearby Amenities ────────────────────────────┐  │
│  │                                                       │  │
│  │   🚇 MRT        Kembangan    0.3 km  (2 within 1km)  │  │
│  │   🛒 Mall       Bedok Mall   0.5 km  (1 within 1km)  │  │
│  │   🍜 Hawker     Blk 85       0.2 km  (5 within 1km)  │  │
│  │   🏫 Primary    XX Primary   0.4 km  (3 within 1km)  │  │
│  │   🏫 Secondary  YY Sec       0.6 km  (2 within 1km)  │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ ROW 3: Comparable Transactions ──────────────────────┐  │
│  │                                                       │  │
│  │   #  Address          Price     Date       Area       │  │
│  │   1  264 Bishan St 24 $585,000  2021-07    104 sqm    │  │
│  │   2   ...                                                │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- **Sidebar** keeps input compact and always visible.
- **Row 1** uses `st.metric()` for the big price number, styled with success color.
- **Row 2** uses `st.columns()` to show amenities in a grid.
- **Row 3** uses `st.dataframe()` or a formatted table for comparable transactions.

The amenity section also serves as a sanity check: users can see distances match
their mental model of the location before trusting the price estimate.

---

## 6. File Structure

```
cs5228_PROJ/
├── app/
│   ├── app.py                    # Streamlit UI (entry point)
│   ├── inference_engine.py       # FeatureBuilder, ModelPredictor, ComparableFinder
│   └── __init__.py
├── Dataset/                      # (existing — unchanged)
│   ├── train.csv
│   ├── train_data_for_modeling(no_standardization).csv
│   ├── features_standard_scaler.pkl
│   ├── knn_feature_stats.pkl
│   ├── numerical_features.json
│   ├── all_final_features.json
│   └── auxiliary-data/
│       ├── sg-hdb-block-details.csv
│       ├── sg-mrt-stations.csv
│       ├── sg-shopping-malls.csv
│       ├── sg-gov-hawkers.csv
│       ├── sg-primary-schools.csv
│       └── sg-secondary-schools.csv
├── Models/
│   └── LightGBM_notebook/
│       ├── lgbm_pipeline_best.joblib
│       └── lgbm_metrics_best.json
├── Utils/                        # (existing — copied tools.py unchanged)
│   ├── tools.py
│   └── ML_training_utils_tools.py
├── Scripts/                      # (existing — unchanged)
├── Plots/                        # (existing — unchanged)
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-08-hdb-valuation-app-design.md
```

### Why `app/` is a package, not a flat script

The inference engine needs to import `tools.py` from `Utils/`. By placing
`app.py` in a sub-package we keep the import paths clean without manipulating
`sys.path` at runtime.

### Runtime dependency

```
# app/__init__.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

All imports in `inference_engine.py` then use project-relative paths:
```python
from Utils.tools import create_auxiliary_location_features
```

### Startup flow

```
app.py imports inference_engine.py
  → __init__ sets PROJECT_ROOT on sys.path
  → inference_engine loads model + data (cached with @st.cache_resource)
  → UI renders
```

---

## 7. Data & Model Loading Strategy

| Artifact | Size (approx.) | Load strategy |
|----------|---------------|---------------|
| `lgbm_pipeline_best.joblib` | ~5 MB | `@st.cache_resource` |
| `features_standard_scaler.pkl` | ~2 KB | `@st.cache_resource` |
| `lgbm_metrics_best.json` | <1 KB | `@st.cache_data` |
| `sg-hdb-block-details.csv` | ~500 KB | `@st.cache_data` |
| 5 auxiliary CSVs (amenities) | ~50 KB total | `@st.cache_data` |
| `train_data_for_modeling(no_standardization).csv` | ~80 MB | `@st.cache_data` |
| KD-Tree index (coordinates) | ~3 MB (in-memory) | Built once inside `@st.cache_resource` |

`@st.cache_resource` ensures model artifacts are loaded exactly once across all
users in a session. Reloading only happens when the file changes (detected via
hash), which is fine for local demo use.

The train data (~80 MB CSV / 162K rows) is the heaviest artifact, but it's
needed for two reasons: (1) KNN reference points, (2) comparable transactions.
In a production setting this would be a database query; for the demo, loading it
into a pandas DataFrame is acceptable.

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| Postal code not found in HDB block details | Show `st.error("Postal code not found. Check and try again.")` |
| Postal code <= 5 digits or NaN | `st.error("Please enter a valid 6-digit postal code.")` |
| floor_area_sqm ≤ 0 or > 300 | `st.warning("Floor area seems unusual. Proceeding, but verify.")` then continue |
| floor_level > MAX_FLOOR | `st.warning("Floor level exceeds the block's max floor.")` then continue |
| Model file not found | `st.error("Model artifact missing.")` — critical, stop |
| Auxiliary CSV not found | `st.error("Data file missing: {filename}")` — critical, stop |
| Prediction fails (unexpected) | `st.exception(e)` — show traceback for debugging |

Errors are user-facing with actionable messages, not raw Python exceptions.

---

## 9. Input Constraints & Validation

| Field | Type | Constraints | Default |
|-------|------|-------------|---------|
| Postal code | str | 6-digit, must exist in hdb-block-details | — (required) |
| Flat type | enum | `2-room`, `3-room`, `4-room`, `5-room`, `executive`, `multi-generation` | `4-room` |
| Flat model | enum | Populated dynamically from train data (23 categories) | `Model A` |
| Floor area (sqm) | float | 30–280 | 90 |
| Floor level (mid) | int | 1–60 | auto: midpoint of floor range from dropdown OR manual input |

**Design note:** Floor level is entered as a **midpoint number** (e.g., 8 for
`07 to 09`), matching how `FLOOR_LEVEL_MID` was computed in training.  Using a
slider or number input is clearer than requiring users to understand floor
range encoding.

---

## 10. Confidence Interval

Because the LightGBM model is a single-point regressor (not a quantile model),
we approximate the interval using the **validation RMSE** from training:

```
interval = [prediction - RMSE_val, prediction + RMSE_val]
```

This is **not** a statistical confidence interval — it's a heuristic using the
model's typical error magnitude.  The UI labels it as "Estimated Range (± RMSE)"
to avoid misleading users into thinking it's a 95% CI.  This is appropriate for
a course demo; a production system would use quantile regression or conformal
prediction.

---

## 11. Testing Strategy

No automated test suite is required for this demo scope.  Manual verification:

1. **Valid postal code, valid inputs** → prediction displayed with all 3 sections
2. **Invalid postal code** → error message shown
3. **Edge-case postal code** (e.g., 018906 — Marina Bay, few HDBs) → verify graceful handling
4. **Extreme floor area** (30 sqm / 280 sqm) → warning shown, prediction still computed
5. **Known HDB transaction** → cross-check prediction against actual resale price
   (sanity check: prediction should be within ±20% of actual)

---

## 12. Known Limitations (documented for report)

| Limitation | Reason | Impact |
|-----------|--------|--------|
| Confidence interval is RMSE-based, not statistical | Point-estimate model | Slight overclaiming if not labeled carefully |
| KNN features depend on full train data | Training leakage if KNN points include test | Not an issue for demo (using train data as reference) |
| No SHAP/feature contribution | Out of scope for v1 | Users can't see *why* the price is what it is |
| Static model (no online learning) | Model trained once | Won't reflect latest market trends |
| Only LightGBM model loaded | XGBoost excluded for simplicity | Could add model selector as future feature |

These are all acceptable for a course demo and documented transparently in
the final report.

---

## 13. Success Criteria

1. `streamlit run app/app.py` starts without errors
2. User enters a known HDB postal code → prediction appears within 3 seconds
3. Amenity distances are plausible (e.g., MRT within 2 km for most HDBs)
4. Comparable transactions are from the same town and flat type
5. RMSE interval is labeled correctly (not "confidence interval")
6. Error states render user-friendly messages, not stack traces

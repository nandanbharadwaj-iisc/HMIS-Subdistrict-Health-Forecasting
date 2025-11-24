import streamlit as st
import pandas as pd
import plotly.express as px
import os, json, shutil

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HMIS Next-Quarter Forecasting Dashboard",
    layout="wide"
)

# Base data directory (relative to this file)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output_dir")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Make sure DATA_DIR exists
os.makedirs(DATA_DIR, exist_ok=True)

files_to_copy = [
    "metrics_v6.csv",
    "best_model_per_target_v6.csv",
    "forecast_timeseries.csv",
]

for fname in files_to_copy:
    src = os.path.join(OUTPUT_DIR, fname)
    dst = os.path.join(DATA_DIR, fname)
    shutil.copy2(src, dst)

# ---------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------
@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    """Cached CSV loader to avoid re-reading on each interaction."""
    return pd.read_csv(csv_path)


def safe_load(csv_name: str) -> pd.DataFrame:
    """
    Safely load a CSV from DATA_DIR.
    Shows a Streamlit warning/error if the file is missing or cannot be read.
    """
    csv_path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(csv_path):
        st.warning(f"Missing data file: {csv_name}")
        return pd.DataFrame()

    try:
        df = load_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Failed to load {csv_name}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# Load core datasets
# ---------------------------------------------------------
model_metrics_df = safe_load("metrics_v6.csv")
best_models_df = safe_load("best_model_per_target_v6.csv")
forecast_timeseries_df = safe_load("forecast_timeseries.csv")

# Basic guardrail – stop if any required CSV is missing or empty
if model_metrics_df.empty or best_models_df.empty or forecast_timeseries_df.empty:
    st.error(
        "One or more required CSVs are missing or empty. "
        "Please ensure metrics_v6.csv, best_model_per_target_v6.csv, "
        "and forecast_timeseries.csv are present in ./data."
    )
    st.stop()

# ---------------------------------------------------------
# Normalize column names (lowercase for consistency)
# ---------------------------------------------------------
model_metrics_df.columns = [c.lower() for c in model_metrics_df.columns]
best_models_df.columns = [c.lower() for c in best_models_df.columns]
forecast_timeseries_df.columns = [c.lower() for c in forecast_timeseries_df.columns]

# ---------------------------------------------------------
# Ensure timeseries schema: derive 'kind'/'value' from label_raw/pred_raw if needed
# ---------------------------------------------------------
if "kind" not in forecast_timeseries_df.columns or "value" not in forecast_timeseries_df.columns:
    has_label = "label_raw" in forecast_timeseries_df.columns
    has_pred = "pred_raw" in forecast_timeseries_df.columns

    if has_label and has_pred:
        # Split into actual vs forecast rows
        base_ts_cols = [
            c for c in forecast_timeseries_df.columns
            if c not in ("label_raw", "pred_raw")
        ]

        actual_df = (
            forecast_timeseries_df[base_ts_cols + ["label_raw"]]
            .rename(columns={"label_raw": "value"})
            .copy()
        )
        actual_df["kind"] = "actual"

        forecast_df = (
            forecast_timeseries_df[base_ts_cols + ["pred_raw"]]
            .rename(columns={"pred_raw": "value"})
            .copy()
        )
        forecast_df["kind"] = "forecast"

        forecast_timeseries_df = pd.concat(
            [actual_df, forecast_df],
            ignore_index=True
        )

    elif has_label:
        # Only actuals present
        forecast_timeseries_df = (
            forecast_timeseries_df.rename(columns={"label_raw": "value"}).copy()
        )
        forecast_timeseries_df["kind"] = "actual"

    elif "value" not in forecast_timeseries_df.columns:
        # Fallback: pick the first numeric column as value
        numeric_cols = forecast_timeseries_df.select_dtypes(include="number").columns
        if len(numeric_cols):
            forecast_timeseries_df = (
                forecast_timeseries_df.rename(columns={numeric_cols[0]: "value"}).copy()
            )
        forecast_timeseries_df["kind"] = "actual"

# Minimal schema sanity check
for required_col in ["year_quarter", "target", "model"]:
    if required_col not in forecast_timeseries_df.columns:
        st.error(
            f"forecast_timeseries.csv is missing required column: {required_col}"
        )
        st.dataframe(forecast_timeseries_df.head())
        st.stop()

# ---------------------------------------------------------
# Data Schema Debug (toggleable)
# ---------------------------------------------------------
with st.expander("🔍 Data Schema Debug", expanded=False):
    st.markdown("**metrics_v6.csv – columns & dtypes**")
    st.write(list(model_metrics_df.columns))
    st.write(model_metrics_df.dtypes)
    st.dataframe(model_metrics_df.head(), use_container_width=True)

    st.markdown("**best_model_per_target_v6.csv – columns & dtypes**")
    st.write(list(best_models_df.columns))
    st.write(best_models_df.dtypes)
    st.dataframe(best_models_df.head(), use_container_width=True)

    st.markdown("**forecast_timeseries.csv – columns & dtypes**")
    st.write(list(forecast_timeseries_df.columns))
    st.write(forecast_timeseries_df.dtypes)
    st.dataframe(forecast_timeseries_df.head(), use_container_width=True)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.title("Filters")

# Target selector
available_targets = sorted(
    forecast_timeseries_df["target"].dropna().unique().tolist()
)
default_target = available_targets[0] if available_targets else None

selected_target = st.sidebar.selectbox(
    "Target Indicator",
    available_targets,
    index=0 if available_targets else None,
    help="Select which HMIS indicator to explore."
)

# Model selector for diagnostics
available_models = sorted(
    model_metrics_df["model"].dropna().unique().tolist()
)
selected_model = st.sidebar.selectbox(
    "Model (for diagnostics)",
    available_models,
    index=0 if available_models else None,
    help="Select a model to inspect its performance."
)

# Map aggregation mode selector
map_value_mode = st.sidebar.selectbox(
    "Map value",
    [
        "forecast (latest quarter)",
        "actual (latest quarter)",
        "forecast (sum all quarters)",
        "actual (sum all quarters)",
    ],
    help="Choose how the state-wise values are aggregated for the spatial heatmap."
)

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
st.title("HMIS Next-Quarter Forecasting Dashboard")

tab_overview, tab_district, tab_models = st.tabs(
    ["Overview", "District Explorer", "Model Performance"]
)

# =========================================================
# Overview Tab
# =========================================================
with tab_overview:
    st.subheader(f"National Trend – {selected_target}")

    # -----------------------------------------------------
    # KPI Cards for best model of selected target
    # -----------------------------------------------------
    target_metrics_df = model_metrics_df[
        model_metrics_df["target"] == selected_target
    ].copy()

    if not target_metrics_df.empty:
        # Pick best row by RMSE for this target
        best_target_row = target_metrics_df.sort_values("rmse").iloc[0]

        kpi_cols = st.columns(4)  # Best Model, RMSE, MAPE, R²
        kpi_cols[0].metric(
            "Best Model",
            str(best_target_row["model"])
        )
        if pd.notna(best_target_row.get("rmse", None)):
            kpi_cols[1].metric(
                "RMSE",
                f"{best_target_row['rmse']:.2f}",
                help="Root Mean Squared Error – lower is better."
            )
        if pd.notna(best_target_row.get("mape", None)):
            kpi_cols[2].metric(
                "MAPE (%)",
                f"{best_target_row['mape']:.1f}",
                help="Mean Absolute Percentage Error."
            )
        if pd.notna(best_target_row.get("r2", None)):
            kpi_cols[3].metric(
                "R²",
                f"{best_target_row['r2']:.3f}",
                help="Coefficient of determination – closer to 1 is better."
            )
    else:
        st.warning(
            "No metrics available for the selected target. "
            "Please verify that metrics_v6.csv contains this target."
        )

    # -----------------------------------------------------
    # National aggregate timeseries (actual vs forecast)
    # -----------------------------------------------------
    target_timeseries_df = forecast_timeseries_df[
        forecast_timeseries_df["target"] == selected_target
    ].copy()

    national_timeseries_df = (
        target_timeseries_df
        .groupby(["year_quarter", "kind"], as_index=False)["value"]
        .sum()
    )

    col_trend, col_spatial = st.columns(2)

    # ---- National line chart ----
    with col_trend:
        st.markdown(
            f"### Actual vs Forecast – {selected_target} (National Aggregate)"
        )

        if national_timeseries_df.empty:
            st.info("No timeseries data available for this target.")
        else:
            national_trend_fig = px.line(
                national_timeseries_df,
                x="year_quarter",
                y="value",
                color="kind",
                markers=True,
                title="",
                labels={
                    "value": "Value",
                    "year_quarter": "Year-Quarter",
                    "kind": "Series Type",
                },
            )
            national_trend_fig.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(national_trend_fig, use_container_width=True)

    # ---- Spatial heatmap of India ----
    with col_spatial:
        st.markdown("### Spatial Heatmap – India")

        india_geojson_path = os.path.join(DATA_DIR, "india_states_geo.json")

        if not os.path.exists(india_geojson_path):
            st.info(
                "To enable the India map, place a GeoJSON file at "
                "`data/india_states_geo.json`.\n\n"
                "If you already have a shapefile/GeoJSON with a different name "
                "or property (e.g., `ST_NM`), rename the file or adjust the "
                "property mapping in the choropleth section of this app."
            )
        else:
            try:
                with open(india_geojson_path, "r", encoding="utf-8") as f:
                    india_geojson = json.load(f)

                # Detect which property key stores state names
                geo_properties = india_geojson["features"][0]["properties"]
                candidate_state_keys = [
                    "state_name",
                    "STATE_NAME",
                    "st_nm",
                    "ST_NM",
                    "NAME_1",
                ]
                state_name_property_key = None
                for prop_key in candidate_state_keys:
                    if prop_key in geo_properties:
                        state_name_property_key = prop_key
                        break

                if state_name_property_key is None:
                    st.error(
                        "Could not find a suitable state-name property in "
                        "india_states_geo.json.\n\n"
                        f"Available properties in first feature: "
                        f"{list(geo_properties.keys())}"
                    )
                else:
                    if "state_name" not in target_timeseries_df.columns:
                        st.error(
                            "`state_name` column not found in "
                            "forecast_timeseries.csv; cannot build map."
                        )
                    else:
                        state_level_df = target_timeseries_df.copy()

                        # Decide which series (actual/forecast) and which quarters to map
                        latest_quarter = state_level_df["year_quarter"].max()
                        is_forecast_row = state_level_df["kind"] == "forecast"
                        is_actual_row = state_level_df["kind"] == "actual"

                        if map_value_mode == "forecast (latest quarter)":
                            mask = is_forecast_row & (
                                state_level_df["year_quarter"] == latest_quarter
                            )
                        elif map_value_mode == "actual (latest quarter)":
                            mask = is_actual_row & (
                                state_level_df["year_quarter"] == latest_quarter
                            )
                        elif map_value_mode == "forecast (sum all quarters)":
                            mask = is_forecast_row
                        else:  # "actual (sum all quarters)"
                            mask = is_actual_row

                        state_level_df = state_level_df[mask]

                        # Aggregate HMIS values by state_name
                        if not state_level_df.empty:
                            state_values_df = (
                                state_level_df.groupby("state_name", as_index=False)[
                                    "value"
                                ].sum()
                            )
                        else:
                            state_values_df = pd.DataFrame(
                                columns=["state_name", "value"]
                            )

                        # Build a full list of states from GeoJSON
                        geo_state_names = [
                            feat["properties"][state_name_property_key]
                            for feat in india_geojson["features"]
                        ]
                        geo_states_df = pd.DataFrame(
                            {"geo_state": geo_state_names}
                        )

                        # Merge HMIS values onto full state list
                        state_values_renamed_df = state_values_df.rename(
                            columns={"state_name": "geo_state"}
                        )
                        choropleth_df = geo_states_df.merge(
                            state_values_renamed_df,
                            on="geo_state",
                            how="left",
                        )
                        choropleth_df["value"] = choropleth_df["value"].fillna(0.0)

                        india_map_fig = px.choropleth(
                            choropleth_df,
                            geojson=india_geojson,
                            locations="geo_state",
                            featureidkey=f"properties.{state_name_property_key}",
                            color="value",
                            hover_name="geo_state",
                            color_continuous_scale="Blues",
                            title="",
                        )
                        india_map_fig.update_geos(
                            fitbounds="locations",
                            visible=False,
                            projection_scale=18,  # zoom in so map fills frame
                            center={"lat": 22.5, "lon": 79.0},  # center of India
                        )
                        india_map_fig.update_layout(
                            height=720,
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(india_map_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading india_states_geo.json: {e}")

    # -----------------------------------------------------
    # Best model per target table
    # -----------------------------------------------------
    st.markdown("### Best Model per Target")
    st.dataframe(
        best_models_df.sort_values(["target", "rmse"]),
        use_container_width=True,
    )

# =========================================================
# District / Subdistrict Explorer
# =========================================================
with tab_district:
    st.subheader("District / Subdistrict Explorer")

    target_timeseries_df = forecast_timeseries_df[
        forecast_timeseries_df["target"] == selected_target
    ].copy()

    # Ensure geographic columns exist
    for geo_col in ["state_name", "district_name", "subdistrict_name"]:
        if geo_col not in target_timeseries_df.columns:
            target_timeseries_df[geo_col] = "Unknown"

    # State selector
    available_states = sorted(
        target_timeseries_df["state_name"].dropna().unique().tolist()
    )
    selected_state = st.selectbox(
        "State",
        available_states,
        index=0 if available_states else None,
    )

    # District selector
    state_filtered_df = target_timeseries_df[
        target_timeseries_df["state_name"] == selected_state
    ].copy()
    available_districts = sorted(
        state_filtered_df["district_name"].dropna().unique().tolist()
    )
    selected_district = st.selectbox(
        "District",
        available_districts,
        index=0 if available_districts else None,
    )

    # Subdistrict selector
    district_filtered_df = state_filtered_df[
        state_filtered_df["district_name"] == selected_district
    ].copy()
    available_subdistricts = sorted(
        district_filtered_df["subdistrict_name"].dropna().unique().tolist()
    )
    selected_subdistrict = st.selectbox(
        "Subdistrict",
        available_subdistricts,
        index=0 if available_subdistricts else None,
    )

    # Filter series for selected location
    location_timeseries_df = district_filtered_df[
        district_filtered_df["subdistrict_name"] == selected_subdistrict
    ].copy()

    if location_timeseries_df.empty:
        st.info("No data for this State / District / Subdistrict combination.")
    else:
        location_line_fig = px.line(
            location_timeseries_df,
            x="year_quarter",
            y="value",
            color="kind",
            markers=True,
            title=(
                f"{selected_target} – "
                f"{selected_state} / {selected_district} / {selected_subdistrict}"
            ),
            labels={
                "value": "Value",
                "year_quarter": "Year-Quarter",
                "kind": "Series Type",
            },
        )
        # 👇 Added more top margin so the title is not cut off
        location_line_fig.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=80, b=0),
        )
        st.plotly_chart(location_line_fig, use_container_width=True)

        st.markdown("Raw timeseries for selected location:")
        st.dataframe(
            location_timeseries_df.sort_values(["year_quarter", "kind"]),
            use_container_width=True,
        )

# =========================================================
# Model Performance Tab
# =========================================================
with tab_models:
    st.subheader("Model Performance Summary")

    # Only keep metrics that actually exist in the metrics CSV
    # (RMSLE and Adjusted R² have been removed from the pipeline)
    metric_options = ["rmse", "mae", "mape", "r2"]
    existing_metric_columns = [
        metric_name
        for metric_name in metric_options
        if metric_name in model_metrics_df.columns
    ]

    if not existing_metric_columns:
        st.warning(
            "No numeric metric columns found in metrics_v6.csv among "
            "['rmse', 'mae', 'mape', 'r2']."
        )
    else:
        # Aggregate average metric per model for comparison
        model_performance_summary_df = (
            model_metrics_df
            .groupby("model")[existing_metric_columns]
            .mean(numeric_only=True)
            .reset_index()
        )

        selected_metric_for_plot = st.selectbox(
            "Metric to plot",
            existing_metric_columns,
            index=0,
            help="Select which evaluation metric to compare across models.",
        )

        model_metric_bar_fig = px.bar(
            model_performance_summary_df,
            x="model",
            y=selected_metric_for_plot,
            title=f"Average {selected_metric_for_plot.upper()} by Model",
            text_auto=True,
            labels={
                "model": "Model",
                selected_metric_for_plot: selected_metric_for_plot.upper(),
            },
        )
        model_metric_bar_fig.update_layout(
            height=600,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(model_metric_bar_fig, use_container_width=True)

        st.markdown("### Full Metrics Table")
        st.dataframe(
            model_metrics_df.sort_values(["target", "model"]),
            use_container_width=True,
        )
    
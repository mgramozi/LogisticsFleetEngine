from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    LATE_THRESHOLD_MIN,
    build_stop_facts,
    delay_attribution,
    driver_scorecard,
    load_tables,
    route_kpis,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data(show_spinner="Loading route and stop data...")
def load_delivery_data(
    raw_dir: str, signature: tuple[tuple[str, int, int], ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_tables(raw_dir)


@st.cache_data(show_spinner="Loading H3 spatial aggregates...")
def load_spatial_data(
    processed_dir: str, signature: tuple[tuple[str, int, int], ...]
) -> pd.DataFrame:
    directory = Path(processed_dir)
    hourly_path = directory / "hex_hourly.parquet"
    if not hourly_path.exists():
        raise FileNotFoundError(
            f"Missing spatial aggregate: {hourly_path.name}. Run `python -m src.etl` from the project root."
        )
    return pd.read_parquet(hourly_path)


def data_signature(
    directory: Path, alternatives: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, int, int], ...]:
    """Create a cache key that changes whenever an input file is replaced."""
    signature = []
    for names in alternatives:
        path = next((directory / name for name in names if (directory / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Expected one of {', '.join(names)} in {directory}")
        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def filter_options(frame: pd.DataFrame, column: str) -> list[str]:
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def main() -> None:
    st.set_page_config(page_title="Fleet Performance", page_icon="🚚", layout="wide")
    st.title("Urban Delivery Fleet Performance")
    st.caption("Dispatch bottlenecks, delivery reliability, and H3 congestion patterns")

    try:
        raw_signature = data_signature(
            RAW_DIR,
            (("routes.parquet", "routes.csv"), ("stops.parquet", "stops.csv")),
        )
        spatial_signature = data_signature(PROCESSED_DIR, (("hex_hourly.parquet",),))
        routes, stops = load_delivery_data(str(RAW_DIR), raw_signature)
        hourly = load_spatial_data(str(PROCESSED_DIR), spatial_signature)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Generate data with `python data/generate_data.py`, then run `python -m src.etl`.")
        st.stop()

    if routes.empty or stops.empty:
        st.error("Route and stop data must both contain at least one record.")
        st.stop()

    st.sidebar.header("Fleet filters")
    selections: dict[str, str] = {}
    for label, column in (
        ("Region", "region"),
        ("Vehicle type", "vehicle_type"),
        ("Depot", "depot"),
        ("Dispatch window", "dispatch_window"),
        ("Routing algorithm", "algorithm"),
        ("Driver", "driver_id"),
    ):
        options = filter_options(routes, column)
        selections[column] = st.sidebar.selectbox(label, ["All", *options])

    late_threshold = st.sidebar.slider(
        "Late delivery threshold (minutes)",
        min_value=0,
        max_value=60,
        value=LATE_THRESHOLD_MIN,
    )
    driver_min_stops = st.sidebar.number_input(
        "Minimum driver stops in both periods",
        min_value=1,
        max_value=500,
        value=50,
        step=5,
    )

    filtered_routes = routes.copy()
    for column, selected in selections.items():
        if selected != "All":
            filtered_routes = filtered_routes.loc[filtered_routes[column].astype(str) == selected]
    if filtered_routes.empty:
        st.warning("No routes match the selected filters.")
        st.stop()

    filtered_stops = stops.loc[stops["route_id"].isin(filtered_routes["route_id"])]
    facts = build_stop_facts(filtered_routes, filtered_stops, threshold=late_threshold)
    route_summary = route_kpis(facts)
    if facts.empty:
        st.warning("No delivery stops match the selected filters.")
        st.stop()

    routes_count = len(route_summary)
    stops_count = len(facts)
    on_time_rate = 1 - facts["is_late"].mean()
    average_wait = filtered_routes["wait_min"].mean()
    first, second, third, fourth = st.columns(4)
    first.metric("Routes", f"{routes_count:,}")
    second.metric("Delivery stops", f"{stops_count:,}")
    third.metric("On-time stops", f"{on_time_rate:.1%}")
    fourth.metric("Average dispatch wait", f"{average_wait:.1f} min")

    left, right = st.columns(2)
    with left:
        depot_metrics = delay_attribution(facts, "depot").reset_index()
        depot_chart = depot_metrics.melt(
            id_vars="depot",
            value_vars=["late_rate", "late_rate_without_wait", "late_due_to_wait_rate"],
            var_name="measure",
            value_name="rate",
        )
        st.plotly_chart(
            px.bar(
                depot_chart,
                x="depot",
                y="rate",
                color="measure",
                barmode="group",
                title="Late-stop rate and dispatch-wait attribution",
                labels={"depot": "Depot", "rate": "Share of stops", "measure": "Measure"},
            ),
            width="stretch",
        )

    with right:
        window_metrics = (
            route_summary.groupby("dispatch_window", observed=True)["late_rate"]
            .mean()
            .rename("mean_late_rate")
            .reset_index()
        )
        st.plotly_chart(
            px.bar(
                window_metrics,
                x="dispatch_window",
                y="mean_late_rate",
                title="Average route late-stop rate by dispatch window",
                labels={"dispatch_window": "Dispatch window", "mean_late_rate": "Mean route late-stop rate"},
            ),
            width="stretch",
        )

    st.subheader("Spatial congestion")
    st.caption("Map cells show H3 zones whose vehicle-specific speed ratio is at or below the selected threshold.")
    dimensions = hourly.copy()
    for column in ("region", "vehicle_type"):
        selected = selections[column]
        if selected != "All":
            dimensions = dimensions.loc[dimensions[column].astype(str) == selected]

    if dimensions.empty:
        st.info("No spatial aggregates match the selected region and vehicle type. Rebuild them with `python -m src.etl`.")
    else:
        available_hours = sorted(dimensions["hour"].dropna().astype(int).unique().tolist())
        map_hour = st.select_slider("Map hour", options=available_hours, value=17 if 17 in available_hours else available_hours[-1])
        max_speed_ratio = st.slider(
            "Maximum speed ratio to show (lower means more congested)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
        )

        congestion_by_hour = (
            dimensions.assign(weighted_speed=dimensions["moving_min"] * dimensions["speed_ratio"])
            .groupby("hour", observed=True)
            .agg(weighted_speed=("weighted_speed", "sum"), moving_min=("moving_min", "sum"))
            .assign(speed_ratio=lambda frame: frame["weighted_speed"] / frame["moving_min"])
            .reset_index()
        )
        st.plotly_chart(
            px.line(
                congestion_by_hour,
                x="hour",
                y="speed_ratio",
                markers=True,
                title="Moving-time-weighted speed ratio by hour",
                labels={"hour": "Hour of day", "speed_ratio": "Speed ratio (1 = free flow)"},
            ),
            width="stretch",
        )

        map_data = dimensions.loc[
            (dimensions["hour"] == map_hour) & (dimensions["speed_ratio"] <= max_speed_ratio)
        ].copy()
        if map_data.empty:
            st.info("No cells meet that congestion threshold for the selected hour.")
        else:
            congestion = (1 - map_data["speed_ratio"].clip(0, 1)).to_numpy()
            map_data["fill_color"] = [
                [int(255 * value), int(200 * (1 - value)), 60, 190]
                for value in congestion
            ]
            layer = pdk.Layer(
                "H3HexagonLayer",
                map_data,
                get_hexagon="h3",
                get_fill_color="fill_color",
                extruded=False,
                pickable=True,
                auto_highlight=True,
            )
            view = pdk.ViewState(
                latitude=float(map_data["hex_lat"].mean()),
                longitude=float(map_data["hex_lon"].mean()),
                zoom=9,
                pitch=30,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    tooltip={
                        "text": (
                            "H3: {h3}\nRegion: {region}\nVehicle: {vehicle_type}\n"
                            "Hour: {hour}\nSpeed ratio: {speed_ratio}\nRoutes: {n_routes}"
                        )
                    },
                ),
                width="stretch",
            )

    st.subheader("Route-level performance")
    st.dataframe(
        route_summary.sort_values("late_rate", ascending=False).head(25),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Driver performance")
    driver_metrics = driver_scorecard(facts, min_stops_per_condition=int(driver_min_stops))
    excluded_drivers = facts["driver_id"].nunique() - len(driver_metrics)
    st.caption(
        f"Peak and off-peak on-time rates are shown for drivers with at least "
        f"{int(driver_min_stops)} stops in each period. {excluded_drivers} driver(s) "
        "do not meet that sample-size threshold."
    )
    if driver_metrics.empty:
        st.warning("No drivers have enough observations in both peak and off-peak periods.")
    else:
        driver_display = driver_metrics.rename(columns={"peak_drop": "peak_drop_pp"})
        driver_display["peak_drop_pp"] *= 100
        st.dataframe(
            driver_display.sort_values("peak_drop_pp", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={
                "on_time_rate": st.column_config.NumberColumn(format="percent"),
                "on_time_peak": st.column_config.NumberColumn(format="percent"),
                "on_time_offpeak": st.column_config.NumberColumn(format="percent"),
                "peak_drop_pp": st.column_config.NumberColumn(
                    "Off-peak minus peak (percentage points)", format="%.1f"
                ),
            },
        )
    st.caption(
        "Data is synthetic. Wait attribution and group comparisons are descriptive and do not establish causation. "
        "Driver comparisons are not adjusted for route assignment or other confounders."
    )


if __name__ == "__main__":
    main()

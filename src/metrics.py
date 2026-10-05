"""
Delivery KPIs, delay decomposition and outlier detection.

Key idea - delay decomposition:
    A stop's delay = dispatch wait at the depot + transit/service delay on the road.
    Because the wait pushes back *every* stop on the route, we can ask a counterfactual:
    "Would this late stop have been on time if the driver had left on schedule?"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LATE_THRESHOLD_MIN = 15
PEAK_WINDOWS = ("morning_peak", "afternoon_peak")
WINDOW_ORDER = ["early", "morning_peak", "midday", "afternoon_peak", "evening"]


# --------------------------------------------------------------------------- loading
def load_tables(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load routes and stops, preferring Parquet and falling back to CSV."""
    raw = Path(raw_dir)
    routes_path = next(
        (raw / name for name in ("routes.parquet", "routes.csv") if (raw / name).is_file()),
        None,
    )
    stops_path = next(
        (raw / name for name in ("stops.parquet", "stops.csv") if (raw / name).is_file()),
        None,
    )
    if routes_path is None or stops_path is None:
        missing = [name for name, path in (("routes", routes_path), ("stops", stops_path)) if path is None]
        raise FileNotFoundError(f"Missing {', '.join(missing)} data file(s) in {raw}")
    routes = (
        pd.read_parquet(routes_path)
        if routes_path.suffix == ".parquet"
        else pd.read_csv(routes_path, parse_dates=["dispatch_planned", "dispatch_actual"])
    )
    stops = (
        pd.read_parquet(stops_path)
        if stops_path.suffix == ".parquet"
        else pd.read_csv(stops_path, parse_dates=["planned_arrival", "actual_arrival"])
    )
    return routes, stops


# --------------------------------------------------------------------------- stop facts
def build_stop_facts(routes: pd.DataFrame, stops: pd.DataFrame,
                     threshold: float = LATE_THRESHOLD_MIN) -> pd.DataFrame:
    """One row per stop, enriched with route attributes and delay decomposition."""
    route_cols = ["route_id", "vehicle_type", "driver_id", "depot", "algorithm",
                  "dispatch_window", "wait_min", "route_km"]
    df = stops.merge(routes[route_cols], on="route_id", how="left", validate="many_to_one")

    df["is_late"] = df["delay_min"] > threshold
    df["transit_delay_min"] = df["delay_min"] - df["wait_min"]
    df["late_without_wait"] = df["transit_delay_min"] > threshold
    df["late_due_to_wait"] = df["is_late"] & ~df["late_without_wait"]
    df["is_peak"] = df["dispatch_window"].isin(PEAK_WINDOWS)
    return df


def delay_attribution(facts: pd.DataFrame, by: str) -> pd.DataFrame:
    """Late rates and descriptive delay decomposition by the selected group."""
    g = facts.groupby(by, observed=True)
    out = pd.DataFrame({
        "stops": g.size(),
        "mean_wait_min": g["wait_min"].mean(),
        "mean_transit_delay_min": g["transit_delay_min"].mean(),
        "late_rate": g["is_late"].mean(),
        "late_rate_without_wait": g["late_without_wait"].mean(),
        "late_due_to_wait_rate": g["late_due_to_wait"].mean(),
    })
    return out.round(3)


def eta_accuracy(facts: pd.DataFrame, by: str | None = None) -> pd.DataFrame | pd.Series:
    """ETA error metrics (actual - planned arrival, minutes)."""
    def _calc(g: pd.DataFrame) -> pd.Series:
        err = g["delay_min"]
        return pd.Series({
            "stops": len(g),
            "bias_min": err.mean(),                 # >0 = systematically late
            "mae_min": err.abs().mean(),
            "p90_abs_error_min": err.abs().quantile(0.9),
            "on_time_rate": 1 - g["is_late"].mean(),
        })
    if by is None:
        return _calc(facts).round(3)
    return facts.groupby(by, observed=True).apply(_calc, include_groups=False).round(3)


# --------------------------------------------------------------------------- route KPIs
def route_kpis(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per route: late rate, delay stats and route attributes."""
    attrs = ["vehicle_type", "driver_id", "depot", "algorithm", "dispatch_window",
             "wait_min", "route_km"]
    g = facts.groupby("route_id")
    out = g[attrs].first()
    out["n_stops"] = g.size()
    out["late_rate"] = g["is_late"].mean()
    out["mean_delay_min"] = g["delay_min"].mean()
    out["max_delay_min"] = g["delay_min"].max()
    out["km_per_stop"] = out["route_km"] / out["n_stops"]
    return out.reset_index()


# --------------------------------------------------------------------------- outliers
def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    lo, hi = iqr_bounds(s, k)
    return (s < lo) | (s > hi)


def flag_outliers_zscore(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=s.index)
    return ((s - s.mean()) / std).abs() > threshold


def flag_outliers_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                           method: str = "iqr", **kwargs) -> pd.Series:
    """Outlier flag computed *within* each group (so a slow depot isn't all 'outliers')."""
    func = flag_outliers_iqr if method == "iqr" else flag_outliers_zscore
    return df.groupby(group_col, observed=True)[value_col].transform(lambda s: func(s, **kwargs))


# --------------------------------------------------------------------------- drivers
def driver_scorecard(facts: pd.DataFrame, min_stops_per_condition: int = 50) -> pd.DataFrame:
    """Compare driver on-time rates in peak and off-peak periods."""
    if min_stops_per_condition < 1:
        raise ValueError("min_stops_per_condition must be at least 1")
    g = facts.groupby("driver_id")
    card = pd.DataFrame({
        "stops": g.size(),
        "on_time_rate": 1 - g["is_late"].mean(),
    })
    for label, mask in (("peak", facts["is_peak"]), ("offpeak", ~facts["is_peak"])):
        sub = facts[mask].groupby("driver_id")
        card[f"stops_{label}"] = sub.size()
        card[f"on_time_{label}"] = 1 - sub["is_late"].mean()

    card = card.dropna()
    card = card[(card["stops_peak"] >= min_stops_per_condition)
                & (card["stops_offpeak"] >= min_stops_per_condition)]
    card["peak_drop"] = card["on_time_offpeak"] - card["on_time_peak"]
    return card.round(3).reset_index()


def driver_clustered_gee(facts: pd.DataFrame):
    """Fit a driver-clustered logistic GEE for adjusted stop lateness comparisons.

    The working correlation accounts for repeated stop outcomes within drivers;
    covariates adjust for route mix but do not turn observational data causal.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    columns = [
        "is_late", "driver_id", "algorithm", "dispatch_window",
        "vehicle_type", "region", "depot", "wait_min", "route_km",
    ]
    missing = sorted(set(columns) - set(facts.columns))
    if missing:
        raise ValueError(f"Missing columns required for driver-clustered analysis: {missing}")
    model_data = facts[columns].dropna().copy()
    if model_data.empty or model_data["is_late"].nunique() < 2:
        raise ValueError("Driver-clustered analysis requires both late and on-time stops.")
    if model_data["driver_id"].nunique() < 2:
        raise ValueError("Driver-clustered analysis requires at least two drivers.")

    return sm.GEE.from_formula(
        "is_late ~ C(algorithm) + C(dispatch_window) + C(vehicle_type) "
        "+ C(region) + C(depot) + wait_min + route_km",
        groups="driver_id",
        data=model_data,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit()
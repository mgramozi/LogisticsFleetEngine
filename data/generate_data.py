"""
Synthetic urban-delivery fleet data generator.

Produces three tables that mimic a real logistics stack:
    routes  - one row per delivery run (vehicle, driver, depot, algorithm, dispatch wait)
    stops   - one row per delivery stop (planned vs actual arrival, delay, late flag)
    pings   - GPS breadcrumbs every ~30 s (lat, lon, speed, status)

Planted "ground truth" effects (so the analysis has real signal to find):
    1. Depot C has a chronic dispatch-wait bottleneck.
    2. Congestion peaks at 07-10h and 15-19h, strongest in the city centre.
    3. 'cluster_nn' stop sequencing beats 'fifo' (shorter routes -> fewer late stops).
    4. Cargo bikes are less congestion-sensitive than vans/trucks.
    5. Wait times have heavy tails (lognormal) -> real outliers for IQR / Z-score.

NOTE: data is simulated. Say so in the README.

Usage:
    python data/generate_data.py --routes 2000 --seed 42 --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CITY_CENTER = (52.5200, 13.4050)  # Berlin; change to any city
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(CITY_CENTER[0]))
ROAD_FACTOR = 1.3  # road distance vs straight line
PING_INTERVAL_MIN = 0.5
LATE_THRESHOLD_MIN = 15
SCHEDULE_PADDING = 1.35  # planned travel time = free-flow * padding

# name: (lat offset deg, lon offset deg, mean wait min, wait sigma)
DEPOTS = {
    "A": (0.060, -0.080, 6.0, 0.45),
    "B": (-0.070, 0.090, 9.0, 0.50),
    "C": (0.040, 0.110, 22.0, 0.65),  # bottleneck depot
}
# region: (lat offset, lon offset, congestion multiplier)
REGIONS = {
    "Center": (0.000, 0.000, 1.6),
    "North": (0.070, 0.010, 0.9),
    "South-East": (-0.050, 0.070, 1.1),
    "West": (0.000, -0.090, 0.8),
}
# vehicle: (free-flow km/h, congestion sensitivity exponent, service minutes)
VEHICLES = {
    "van": (32.0, 1.0, 4.0),
    "truck": (26.0, 1.15, 6.0),
    "cargo_bike": (16.0, 0.35, 3.0),
}
VEHICLE_MIX = [0.55, 0.15, 0.30]
# window: (start hour, length hours, share of routes)
WINDOWS = {
    "early": (5.0, 2.0, 0.15),
    "morning_peak": (7.0, 3.0, 0.25),
    "midday": (10.0, 5.0, 0.25),
    "afternoon_peak": (15.0, 4.0, 0.25),
    "evening": (19.0, 2.0, 0.10),
}
ALGORITHMS = ["fifo", "cluster_nn"]


def congestion_index(hour: float, region_mult: float) -> float:
    """Multiplier >= 1 applied to free-flow travel time."""
    base = (
        0.15
        + 0.55 * np.exp(-(((hour - 8.5) / 1.3) ** 2))
        + 0.65 * np.exp(-(((hour - 17.0) / 1.6) ** 2))
    )
    return 1.0 + base * region_mult


def dist_km(p: np.ndarray, q: np.ndarray) -> float:
    dlat = (p[0] - q[0]) * KM_PER_DEG_LAT
    dlon = (p[1] - q[1]) * KM_PER_DEG_LON
    return float(np.hypot(dlat, dlon)) * ROAD_FACTOR


def greedy_order(depot: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sequencing starting from the depot."""
    remaining = list(range(len(stops)))
    order, pos = [], depot
    while remaining:
        nxt = min(remaining, key=lambda i: dist_km(pos, stops[i]))
        order.append(nxt)
        pos = stops[nxt]
        remaining.remove(nxt)
    return np.array(order)


def generate(n_routes: int, seed: int, n_days: int = 60, n_drivers: int = 120):
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2025-03-03")
    driver_skill = rng.normal(1.0, 0.07, n_drivers).clip(0.8, 1.25)

    region_names = list(REGIONS)
    window_names = list(WINDOWS)
    window_p = [WINDOWS[w][2] for w in window_names]
    vehicle_names = list(VEHICLES)

    routes, stops_rows, ping_chunks = [], [], []

    for r in range(n_routes):
        route_id = f"R{r:06d}"
        depot_name = rng.choice(list(DEPOTS))
        d_lat, d_lon, wait_mean, wait_sigma = DEPOTS[depot_name]
        depot = np.array([CITY_CENTER[0] + d_lat, CITY_CENTER[1] + d_lon])

        vehicle = rng.choice(vehicle_names, p=VEHICLE_MIX)
        speed, sens, service_plan = VEHICLES[vehicle]
        driver_idx = int(rng.integers(n_drivers))
        skill = driver_skill[driver_idx]
        algorithm = rng.choice(ALGORITHMS)
        window = rng.choice(window_names, p=window_p)
        w_start, w_len, _ = WINDOWS[window]
        day = start_date + pd.Timedelta(days=int(rng.integers(n_days)))

        # dispatch time (minutes since midnight)
        t_plan = (w_start + rng.uniform(0, w_len)) * 60
        dispatch_planned = t_plan
        wait = float(rng.lognormal(np.log(wait_mean) - wait_sigma**2 / 2, wait_sigma))
        t_act = t_plan + wait

        # stops clustered inside one region
        region = rng.choice(region_names)
        r_lat, r_lon, r_mult = REGIONS[region]
        n_stops = int(rng.integers(8, 21))
        centre = np.array([CITY_CENTER[0] + r_lat, CITY_CENTER[1] + r_lon])
        stop_xy = centre + rng.normal(0, 0.012, (n_stops, 2))
        order = (
            greedy_order(depot, stop_xy) if algorithm == "cluster_nn" else rng.permutation(n_stops)
        )
        stop_xy = stop_xy[order]

        # waiting pings at depot
        n_wait = max(1, int(wait / PING_INTERVAL_MIN))
        wt = np.linspace(dispatch_planned, t_act, n_wait, endpoint=False)
        pings = [(wt, np.full(n_wait, depot[0]), np.full(n_wait, depot[1]), np.zeros(n_wait), 0)]

        pos, total_km, late_count = depot, 0.0, 0
        for s, xy in enumerate(stop_xy):
            d = dist_km(pos, xy)
            total_km += d
            free_tt = d / speed * 60
            plan_tt = free_tt * SCHEDULE_PADDING  # dispatcher assumes typical traffic
            cong = congestion_index(t_act / 60, r_mult) ** sens
            act_tt = free_tt * cong * rng.lognormal(0, 0.12) * (2 - skill)
            act_tt = max(act_tt, 0.3)

            # moving pings
            n_p = max(1, int(act_tt / PING_INTERVAL_MIN))
            frac = np.linspace(0, 1, n_p, endpoint=False)[:, None]
            path = pos + (xy - pos) * frac + rng.normal(0, 8e-5, (n_p, 2))
            times = t_act + frac[:, 0] * act_tt
            spd = np.full(n_p, d / act_tt * 60) * rng.uniform(0.7, 1.3, n_p)
            pings.append((times, path[:, 0], path[:, 1], spd, 1))

            t_plan += plan_tt
            t_act += act_tt
            planned_arrival, actual_arrival = t_plan + 10, t_act  # 10 min scheduled buffer
            delay = actual_arrival - planned_arrival
            is_late = delay > LATE_THRESHOLD_MIN
            late_count += is_late

            service_act = service_plan * rng.lognormal(0, 0.3) * (2 - skill)
            stops_rows.append(
                (
                    route_id, s + 1, xy[0], xy[1], region,
                    day + pd.Timedelta(minutes=planned_arrival),
                    day + pd.Timedelta(minutes=actual_arrival),
                    delay, bool(is_late), service_act,
                )
            )

            # servicing pings (stationary)
            n_s = max(1, int(service_act / PING_INTERVAL_MIN))
            st = t_act + np.linspace(0, service_act, n_s, endpoint=False)
            pings.append((st, np.full(n_s, xy[0]), np.full(n_s, xy[1]), np.zeros(n_s), 2))

            t_plan += service_plan
            t_act += service_act
            pos = xy

        routes.append(
            (
                route_id, f"V{rng.integers(1, 81):03d}", vehicle, f"D{driver_idx:03d}",
                depot_name, algorithm, window, region,
                day + pd.Timedelta(minutes=dispatch_planned),
                day + pd.Timedelta(minutes=dispatch_planned + wait),
                wait, n_stops, total_km, late_count / n_stops,
            )
        )

        for times, lat, lon, spd, status in pings:
            ping_chunks.append(
                pd.DataFrame(
                    {
                        "route_id": route_id,
                        "ts": day + pd.to_timedelta(times, unit="min"),
                        "lat": lat,
                        "lon": lon,
                        "speed_kmh": spd,
                        "status": status,  # 0 waiting, 1 moving, 2 servicing
                    }
                )
            )

    routes_df = pd.DataFrame(
        routes,
        columns=[
            "route_id", "vehicle_id", "vehicle_type", "driver_id", "depot", "algorithm",
            "dispatch_window", "region", "dispatch_planned", "dispatch_actual",
            "wait_min", "n_stops", "route_km", "late_rate",
        ],
    )
    stops_df = pd.DataFrame(
        stops_rows,
        columns=[
            "route_id", "stop_seq", "lat", "lon", "region", "planned_arrival",
            "actual_arrival", "delay_min", "is_late", "service_min",
        ],
    )
    pings_df = pd.concat(ping_chunks, ignore_index=True)
    return routes_df, stops_df, pings_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    routes, stops, pings = generate(args.routes, args.seed)

    try:
        routes.to_parquet(out / "routes.parquet", index=False)
        stops.to_parquet(out / "stops.parquet", index=False)
        pings.to_parquet(out / "pings.parquet", index=False)
        fmt = "parquet"
    except ImportError:  # pyarrow missing
        for name in ("routes.parquet", "stops.parquet", "pings.parquet"):
            (out / name).unlink(missing_ok=True)
        routes.to_csv(out / "routes.csv", index=False)
        stops.to_csv(out / "stops.csv", index=False)
        pings.to_csv(out / "pings.csv.gz", index=False)
        fmt = "csv"

    print(f"Saved ({fmt}) to {out}/")
    print(f"  routes: {len(routes):,} | stops: {len(stops):,} | pings: {len(pings):,}")
    print(f"  overall late-stop rate: {stops.is_late.mean():.1%}")


if __name__ == "__main__":
    main()
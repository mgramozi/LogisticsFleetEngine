
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterator

import duckdb
import h3
import pandas as pd
import pyarrow.parquet as pq

H3_RES = 8
PING_INTERVAL_MIN = 0.5
MAX_PING_GAP_MIN = 1.5
PING_BATCH_SIZE = 100_000
MIN_PINGS_PER_HEX_HOUR = 20

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WAITING, MOVING, SERVICING = 0, 1, 2
PING_COLUMNS = ["route_id", "ts", "lat", "lon", "speed_kmh", "status"]
ROUTE_COLUMNS = ["route_id", "vehicle_type", "region"]


def add_h3_index(df: pd.DataFrame, res: int = H3_RES,
                 lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Return a copy of df with an 'h3' column (cell id as string)."""
    out = df.copy()
    out["h3"] = [
        h3.latlng_to_cell(lat, lon, res)
        for lat, lon in zip(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    ]
    return out


def add_hex_centers(df: pd.DataFrame, h3_col: str = "h3") -> pd.DataFrame:
    """Add 'hex_lat' / 'hex_lon' (cell centroid) for plotting."""
    out = df.copy()
    centers = {cell: h3.cell_to_latlng(cell) for cell in out[h3_col].unique()}
    out["hex_lat"] = out[h3_col].map(lambda c: centers[c][0])
    out["hex_lon"] = out[h3_col].map(lambda c: centers[c][1])
    return out


def _find_table(raw_dir: Path, name: str) -> Path:
    for suffix in (".parquet", ".csv", ".csv.gz"):
        path = raw_dir / f"{name}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {name}.parquet, {name}.csv, or {name}.csv.gz in {raw_dir}"
    )


def _read_batches(path: Path) -> Iterator[pd.DataFrame]:
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=PING_BATCH_SIZE, columns=PING_COLUMNS):
            yield batch.to_pandas()
        return

    for batch in pd.read_csv(path, usecols=PING_COLUMNS, chunksize=PING_BATCH_SIZE):
        batch["ts"] = pd.to_datetime(batch["ts"], errors="raise")
        yield batch


def load_pings_with_h3(con: duckdb.DuckDBPyConnection,
                       raw_dir: Path = RAW_DIR, res: int = H3_RES) -> int:
    """Stream raw pings, tag them with H3, and stage them in DuckDB."""
    raw_dir = Path(raw_dir)
    pings_path = _find_table(raw_dir, "pings")
    routes_path = _find_table(raw_dir, "routes")
    if routes_path.suffix == ".parquet":
        routes = pd.read_parquet(routes_path, columns=ROUTE_COLUMNS)
    else:
        routes = pd.read_csv(routes_path, usecols=ROUTE_COLUMNS)
    if routes["route_id"].duplicated().any():
        raise ValueError(f"Route IDs must be unique in {routes_path}")
    route_lookup = routes.set_index("route_id")[["vehicle_type", "region"]]

    con.execute("DROP TABLE IF EXISTS pings_h3")
    total = 0
    for batch in _read_batches(pings_path):
        unknown_routes = ~batch["route_id"].isin(route_lookup.index)
        if unknown_routes.any():
            examples = batch.loc[unknown_routes, "route_id"].drop_duplicates().head(5).tolist()
            raise ValueError(f"GPS pings reference unknown route IDs: {examples}")
        batch = batch.join(route_lookup, on="route_id", validate="many_to_one")
        batch = add_h3_index(batch, res)
        con.register("ping_batch", batch)
        try:
            if total == 0:
                con.execute("CREATE TABLE pings_h3 AS SELECT * FROM ping_batch")
            else:
                con.execute("INSERT INTO pings_h3 SELECT * FROM ping_batch")
        finally:
            con.unregister("ping_batch")
        total += len(batch)

    if total == 0:
        raise ValueError(f"No GPS pings found in {pings_path}")
    return total


_ACTIVITY_CTES = f"""
WITH sequenced AS (
    SELECT *,
           lead(ts) OVER (PARTITION BY route_id ORDER BY ts) AS next_ts
    FROM pings_h3
),
intervals AS (
    SELECT *,
           CASE WHEN next_ts IS NULL THEN NULL ELSE least(
                   greatest(date_diff('microseconds', ts, next_ts) / 60000000.0, 0),
                   {MAX_PING_GAP_MIN}
               )
           END AS duration_min
    FROM sequenced
)
"""

HEX_HOURLY_SQL = f"""
CREATE OR REPLACE TABLE hex_hourly AS
{_ACTIVITY_CTES},
free_flow AS (
    SELECT vehicle_type, quantile_cont(speed_kmh, 0.9) AS free_flow_kmh
    FROM intervals
    WHERE status = {MOVING}
    GROUP BY vehicle_type
)
SELECT
    p.h3,
    p.region,
    p.vehicle_type,
    hour(p.ts) AS hour,
    count(*) AS n_pings,
    count(DISTINCT p.route_id) AS n_routes,
    avg(p.speed_kmh) FILTER (WHERE p.status = {MOVING}) AS avg_moving_speed_kmh,
    sum(p.duration_min) FILTER (WHERE p.status = {WAITING}) AS waiting_min,
    sum(p.duration_min) FILTER (WHERE p.status = {SERVICING}) AS servicing_min,
    sum(p.duration_min) FILTER (WHERE p.status = {MOVING}) AS moving_min,
    avg(p.speed_kmh / f.free_flow_kmh) FILTER (WHERE p.status = {MOVING}) AS speed_ratio
FROM intervals p
JOIN free_flow f USING (vehicle_type)
GROUP BY p.h3, p.region, p.vehicle_type, hour(p.ts)
HAVING count(*) >= {MIN_PINGS_PER_HEX_HOUR}
   AND count(*) FILTER (WHERE p.status = {MOVING}) >= {MIN_PINGS_PER_HEX_HOUR}
"""

HEX_SUMMARY_SQL = f"""
CREATE OR REPLACE TABLE hex_summary AS
{_ACTIVITY_CTES}
SELECT
    h3,
    region,
    vehicle_type,
    count(*) AS n_pings,
    count(DISTINCT route_id) AS n_routes,
    avg(speed_kmh) FILTER (WHERE status = {MOVING}) AS avg_moving_speed_kmh,
    sum(duration_min) FILTER (WHERE status = {WAITING}) AS waiting_min,
    sum(duration_min) FILTER (WHERE status = {SERVICING}) AS servicing_min,
    sum(duration_min) FILTER (WHERE status = {MOVING}) AS moving_min
FROM intervals
GROUP BY h3, region, vehicle_type
"""


def build_hex_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the aggregations; return (hex_hourly, hex_summary) with hex centres."""
    con.execute(HEX_HOURLY_SQL)
    con.execute(HEX_SUMMARY_SQL)
    hourly = add_hex_centers(con.execute("SELECT * FROM hex_hourly").df())
    summary = add_hex_centers(con.execute("SELECT * FROM hex_summary").df())
    return hourly, summary


def run_pipeline(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
                 res: int = H3_RES) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fleet-etl-") as temp_dir:
        con = duckdb.connect(str(Path(temp_dir) / "staging.duckdb"))
        try:
            n = load_pings_with_h3(con, raw_dir, res)
            print(f"Indexed {n:,} pings into H3 res {res}")
            hourly, summary = build_hex_tables(con)
        finally:
            con.close()

    hourly.to_parquet(out_dir / "hex_hourly.parquet", index=False)
    summary.to_parquet(out_dir / "hex_summary.parquet", index=False)
    print(f"hex_hourly: {len(hourly):,} rows | hex_summary: {len(summary):,} hexagons")
    return hourly, summary


if __name__ == "__main__":
    hourly, summary = run_pipeline()
    depot_summary = summary.groupby("h3", as_index=False).agg(
        waiting_min=("waiting_min", "sum"),
        n_routes=("n_routes", "sum"),
        hex_lat=("hex_lat", "first"),
        hex_lon=("hex_lon", "first"),
    )
    print("\nTop 5 hexagons by waiting time (depot bottleneck candidates):")
    print(
        depot_summary.loc[depot_summary["waiting_min"] > 0]
        .nlargest(5, "waiting_min")[["h3", "waiting_min", "n_routes", "hex_lat", "hex_lon"]]
    )
    print("\nSlowest 5 hex-hours (congestion candidates):")
    print(
        hourly.nsmallest(5, "speed_ratio")[
            ["h3", "region", "vehicle_type", "hour", "speed_ratio", "avg_moving_speed_kmh"]
        ]
    )

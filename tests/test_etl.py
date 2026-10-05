import h3
import pandas as pd

from src.etl import add_hex_centers, add_h3_index, run_pipeline


def test_add_h3_index_and_centers_preserves_input() -> None:
    pings = pd.DataFrame({"lat": [52.52, 52.521], "lon": [13.405, 13.406]})

    indexed = add_h3_index(pings)
    result = add_hex_centers(indexed)

    assert "h3" not in pings.columns
    assert indexed["h3"].tolist() == [
        h3.latlng_to_cell(52.52, 13.405, 8),
        h3.latlng_to_cell(52.521, 13.406, 8),
    ]
    assert result["hex_lat"].between(52.4, 52.7).all()
    assert result["hex_lon"].between(13.2, 13.6).all()


def test_run_pipeline_writes_region_and_vehicle_aggregates(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    routes = pd.DataFrame(
        {
            "route_id": ["route-1"],
            "vehicle_type": ["van"],
            "region": ["Center"],
        }
    )
    pings = pd.DataFrame(
        {
            "route_id": ["route-1"] * 26,
            "ts": pd.date_range("2025-03-03 08:00:00", periods=26, freq="30s"),
            "lat": [52.52] * 26,
            "lon": [13.405] * 26,
            "speed_kmh": [0.0, *([24.0] * 25)],
            "status": [0, *([1] * 25)],
        }
    )
    routes.to_parquet(raw_dir / "routes.parquet", index=False)
    pings.to_parquet(raw_dir / "pings.parquet", index=False)

    hourly, summary = run_pipeline(raw_dir=raw_dir, out_dir=processed_dir)

    assert len(hourly) == 1
    assert hourly.loc[0, "region"] == "Center"
    assert hourly.loc[0, "vehicle_type"] == "van"
    assert hourly.loc[0, "n_routes"] == 1
    assert hourly.loc[0, "moving_min"] == 12.0
    assert len(summary) == 1
    assert summary.loc[0, "waiting_min"] == 0.5
    assert (processed_dir / "hex_hourly.parquet").is_file()
    assert (processed_dir / "hex_summary.parquet").is_file()

    saved_hourly = pd.read_parquet(processed_dir / "hex_hourly.parquet")
    assert saved_hourly[["region", "vehicle_type"]].iloc[0].tolist() == ["Center", "van"]


def test_run_pipeline_supports_csv_and_uses_elapsed_ping_time(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "route_id": ["route-1"],
            "vehicle_type": ["van"],
            "region": ["Center"],
        }
    ).to_csv(raw_dir / "routes.csv", index=False)
    pings = pd.DataFrame(
        {
            "route_id": ["route-1"] * 22,
            "ts": pd.date_range("2025-03-03 08:00:00", periods=22, freq="15s"),
            "lat": [52.52] * 22,
            "lon": [13.405] * 22,
            "speed_kmh": [20.0] * 22,
            "status": [1] * 22,
        }
    )
    pings.to_csv(raw_dir / "pings.csv.gz", index=False, compression="gzip")

    hourly, summary = run_pipeline(raw_dir=raw_dir, out_dir=processed_dir)

    assert len(hourly) == 1
    assert hourly.loc[0, "moving_min"] == 5.25
    assert summary.loc[0, "moving_min"] == 5.25

import numpy as np
import pandas as pd
import pytest

from src.metrics import driver_clustered_gee, driver_scorecard, load_tables


def test_load_tables_falls_back_to_csv(tmp_path) -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["route-1"],
            "vehicle_type": ["van"],
            "driver_id": ["driver-1"],
            "depot": ["A"],
            "algorithm": ["fifo"],
            "dispatch_window": ["early"],
            "wait_min": [4.0],
            "route_km": [12.0],
            "dispatch_planned": ["2025-03-03 08:00:00"],
            "dispatch_actual": ["2025-03-03 08:04:00"],
        }
    )
    stops = pd.DataFrame(
        {
            "route_id": ["route-1"],
            "planned_arrival": ["2025-03-03 08:30:00"],
            "actual_arrival": ["2025-03-03 08:35:00"],
            "delay_min": [5.0],
        }
    )
    routes.to_csv(tmp_path / "routes.csv", index=False)
    stops.to_csv(tmp_path / "stops.csv", index=False)

    loaded_routes, loaded_stops = load_tables(tmp_path)

    assert pd.api.types.is_datetime64_any_dtype(loaded_routes["dispatch_planned"])
    assert pd.api.types.is_datetime64_any_dtype(loaded_stops["planned_arrival"])
    assert loaded_routes.loc[0, "route_id"] == "route-1"


def test_driver_scorecard_requires_minimum_observations_in_both_periods() -> None:
    facts = pd.DataFrame(
        {
            "driver_id": ["driver-1"] * 4 + ["driver-2"] * 3,
            "is_late": [False, True, False, False, False, False, True],
            "is_peak": [True, True, False, False, True, False, False],
        }
    )

    scorecard = driver_scorecard(facts, min_stops_per_condition=2)

    assert scorecard["driver_id"].tolist() == ["driver-1"]
    assert scorecard.loc[0, "stops_peak"] == 2
    assert scorecard.loc[0, "stops_offpeak"] == 2
    assert scorecard.loc[0, "peak_drop"] == 0.5
    with pytest.raises(ValueError, match="at least 1"):
        driver_scorecard(facts, min_stops_per_condition=0)


def test_driver_clustered_gee_accounts_for_repeated_driver_observations() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for driver in range(12):
        for algorithm in ("fifo", "cluster_nn"):
            for window in ("early", "morning_peak"):
                for repeat in range(2):
                    rows.append(
                        {
                            "driver_id": f"driver-{driver}",
                            "algorithm": algorithm,
                            "dispatch_window": window,
                            "vehicle_type": rng.choice(["van", "truck", "cargo_bike"]),
                            "region": rng.choice(["Center", "West", "North", "South-East"]),
                            "depot": rng.choice(["A", "B", "C"]),
                            "wait_min": rng.uniform(2, 30),
                            "route_km": rng.uniform(5, 40),
                            "is_late": rng.random() < (
                                0.4 if algorithm == "fifo" else 0.2
                            ),
                        }
                    )
    facts = pd.DataFrame(rows)

    result = driver_clustered_gee(facts)

    assert "C(algorithm)[T.fifo]" in result.params.index
    assert result.model.cov_struct.__class__.__name__ == "Exchangeable"
    assert result.conf_int().shape[1] == 2

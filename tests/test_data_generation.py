import sys

import pandas as pd

from data.generate_data import main


def test_generator_csv_fallback_removes_stale_parquet_files(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    for name in ("routes.parquet", "stops.parquet", "pings.parquet"):
        (out_dir / name).write_bytes(b"stale")

    def parquet_unavailable(self, *args, **kwargs):
        raise ImportError("Parquet writer unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", parquet_unavailable)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_data.py", "--routes", "2", "--seed", "5", "--out", str(out_dir)],
    )

    main()

    assert (out_dir / "routes.csv").is_file()
    assert (out_dir / "stops.csv").is_file()
    assert (out_dir / "pings.csv.gz").is_file()
    assert not any((out_dir / name).exists() for name in (
        "routes.parquet", "stops.parquet", "pings.parquet"
    ))
    assert not pd.read_csv(out_dir / "pings.csv.gz").empty

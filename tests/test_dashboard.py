from app.main import data_signature


def test_data_signature_changes_when_input_file_changes(tmp_path) -> None:
    routes = tmp_path / "routes.csv"
    stops = tmp_path / "stops.csv"
    routes.write_text("route_id\nR1\n", encoding="utf-8")
    stops.write_text("route_id\nR1\n", encoding="utf-8")
    alternatives = (("routes.parquet", "routes.csv"), ("stops.parquet", "stops.csv"))

    first_signature = data_signature(tmp_path, alternatives)
    routes.write_text("route_id\nR1\nR2\n", encoding="utf-8")
    second_signature = data_signature(tmp_path, alternatives)

    assert first_signature != second_signature

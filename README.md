# Logistics and Delivery Fleet Performance Engine

The Logistics and Delivery Fleet Performance Engine turns delivery routes, stop records, and GPS pings into practical views of fleet reliability. It is designed to help an operations team ask questions such as where vehicles spend time waiting, when and where traffic slows them down, how delivery performance changes across dispatch periods, and whether those patterns differ between drivers or routing methods.

The project is an end-to-end analytical prototype. It includes a synthetic Berlin data generator, a spatial-temporal processing pipeline, statistical notebooks, and an interactive Streamlit dashboard. The supplied data is simulated and deliberately contains known patterns so that the analysis can be demonstrated and checked. It is not evidence about a real delivery operation, and results from it must not be presented as real-world findings.

## How the project works

The workflow begins with route, stop, and GPS ping files. The ETL reads pings in bounded batches, joins each ping to its route attributes, assigns an Uber H3 cell, and stages the records in a temporary disk-backed DuckDB database. DuckDB then calculates hourly and overall aggregates by H3 cell, region, and vehicle type. The notebooks explore those outputs and assess delivery delays, while the dashboard provides interactive filters and map views for fleet managers.

H3 is a hierarchical spatial index. In this project, resolution 8 divides the city into small hexagonal zones suitable for exploring urban delivery patterns. DuckDB performs the large-table aggregations, while H3 indexing is applied to each incoming batch in Python. The temporary database is removed when processing finishes; only the compact Parquet aggregates are retained.

## What is included

| Area | Purpose |
| --- | --- |
| `data/generate_data.py` | Creates reproducible synthetic routes, delivery stops, and GPS pings. |
| `src/etl.py` | Streams Parquet or CSV ping data, indexes locations to H3, and builds spatial aggregates. |
| `src/metrics.py` | Provides delivery, delay-attribution, driver scorecard, outlier, and driver-clustered statistical metrics. |
| `notebooks/01_spatial_aggregation.ipynb` | Examines depot waits, hourly traffic speeds, and H3 congestion maps. |
| `notebooks/02_bottleneck_statistical_testing.ipynb` | Investigates delay attribution, wait-time outliers, and differences across algorithms and dispatch windows. |
| `app/main.py` | Runs the Streamlit fleet operations dashboard. |
| `tests/` | Contains automated checks for the ETL and analytical helpers. |

Generated raw data and spatial aggregates are intentionally excluded from version control. The application and notebooks read them from `data/raw` and `data/processed`.

## Requirements and installation

Use Python 3.10 or later. Run the following commands in PowerShell from the project directory to create an isolated environment and install the project with its notebook and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The repository also includes `requirements.txt` for a direct dependency installation:

```powershell
python -m pip install -r requirements.txt
```

Parquet is the preferred storage format. If PyArrow is unavailable to the generator, it writes CSV route and stop files and a compressed CSV ping file instead. The ETL and analysis helpers support those CSV outputs.

## Generate data and build the aggregates

From the project directory, generate the default dataset with a fixed random seed:

```powershell
python data\generate_data.py --routes 2000 --seed 42 --out data\raw
```

The generator creates 2,000 routes, roughly 28,000 delivery stops, and hundreds of thousands of GPS pings. The exact row counts depend on the seed and simulation settings. The default data is intended for demonstrations and local development, not as a production-scale benchmark.

Each route record identifies its driver, vehicle, depot, dispatch window, routing algorithm, dispatch wait, and route distance. Each stop record contains its planned and actual arrival times, calculated delay, and service duration. Each GPS ping contains a route ID, timestamp, latitude, longitude, speed, and activity status; the status values are `0` for dispatch waiting, `1` for vehicle movement, and `2` for stop servicing. The ETL also requires route records to provide a vehicle type and service region so each ping can be analyzed in context.

Build the spatial aggregates after the raw files have been generated:

```powershell
python -m src.etl
```

The pipeline writes `data/processed/hex_hourly.parquet` and `data/processed/hex_summary.parquet`. It reads source files in batches of 100,000 pings, so it does not require loading the entire raw ping table into a pandas DataFrame at once. DuckDB uses a temporary on-disk database for staging and aggregation.

Activity minutes are estimated from the elapsed time between successive GPS pings belonging to the same route. An interval is capped at 1.5 minutes to avoid treating long gaps as continuous activity. Each interval is attributed to the preceding ping's H3 cell and hour; intervals are not split at cell or hour boundaries. As a result, these measures are sampled operational estimates, not exact time accounting. The hourly table filters out groups with fewer than 20 pings or fewer than 20 moving pings to reduce the effect of sparse observations.

When regenerating data, run the ETL again so that the spatial outputs correspond to the new raw files. The dashboard detects changes to its source files and refreshes cached data when their modification time or size changes.

## Explore the analysis

Launch Jupyter from the project directory with:

```powershell
python -m jupyter notebook
```

The first notebook, `notebooks/01_spatial_aggregation.ipynb`, summarizes dispatch waits from route records, plots moving-time-weighted speed by hour, and writes an interactive H3 map to `data/processed/fleet_map.html`.

The second notebook, `notebooks/02_bottleneck_statistical_testing.ipynb`, compares lateness before and after subtracting the route's dispatch wait, flags unusually long waits, and compares route-level late-stop rates across routing algorithms and dispatch windows. Mann–Whitney U and Kruskal–Wallis tests provide unadjusted distribution comparisons, with Holm correction for the pairwise window tests. The notebook also reports a logistic generalized estimating equation (GEE) that accounts for repeated stop observations from the same driver and adjusts for available route characteristics.

The rank tests are useful exploratory summaries, not replacements for the driver-clustered model or a controlled experiment. The GEE also remains observational: it cannot remove unmeasured confounding or prove that a routing algorithm or dispatch time caused a change in delivery performance. Review effect sizes, sample sizes, uncertainty, route assignment, and operational context before making decisions.

## Use the fleet dashboard

Start the interactive dashboard with:

```powershell
streamlit run app\main.py
```

The sidebar filters the route and stop metrics by region, vehicle type, depot, dispatch window, routing algorithm, and driver. It also allows the lateness threshold and minimum driver sample size to be changed. The main view presents route and stop totals, on-time performance, average dispatch wait, depot delay attribution, dispatch-window comparisons, an hourly traffic trend, and an H3 congestion map.

The driver section compares on-time performance during peak and off-peak dispatch windows. A driver is shown only when the selected minimum number of stops is available in both periods. This reduces attention to unstable small-sample rates; it does not make driver comparisons fair or causal. Differences in route difficulty, vehicle, geography, and assignment can still affect the results.

The `peak_drop` score in the analytical helper is the off-peak on-time rate minus the peak on-time rate. In the dashboard it is displayed in percentage points, so a positive number means the driver's observed on-time rate was higher off-peak.

## Definitions and interpretation

A stop is counted as late when its delay exceeds the selected threshold, which defaults to 15 minutes. The dashboard's threshold can be changed without rebuilding the aggregates.

`late_due_to_wait_rate` is the share of stops that are late in the observed data but would fall below the late threshold after subtracting their route's dispatch wait. `late_rate_without_wait` is the share that remains late after that subtraction. These are descriptive counterfactual calculations. Dispatch wait can affect every stop on a route, but subtracting it does not establish what would actually have happened if the route had departed earlier.

The spatial `speed_ratio` compares moving speed with the 90th-percentile moving speed for the same vehicle type in the supplied dataset. Lower values indicate slower observed movement relative to that reference. This is a dataset-derived reference, not a validated speed limit or an external measurement of free-flow traffic.

Outlier flags identify observations that merit investigation. A statistical outlier is not necessarily an error or a driver-performance problem. Compare global and within-depot flags in context, and check data quality before acting.

## Validate the project

Run the automated tests from the project directory:

```powershell
python -m pytest
```

The tests use temporary directories and fixtures; they do not replace generated project data. The dashboard can be smoke-tested locally by running the Streamlit command and checking that its metrics, charts, and map load with the generated files.

## Current scope and next steps

This implementation is a strong starting point for exploratory fleet analytics, but it is not yet a production logistics platform. The generator is synthetic, the sampled-time assignment is approximate, and the dashboard currently visualizes congestion rather than calculating optimized routes. A production version would need representative GPS and dispatch data, data-quality and coordinate validation, explicit timezone and service-calendar handling, route-aware interval splitting, scalable monitoring and persistence, and an evaluation design agreed with operations stakeholders.

For reliable driver or algorithm comparisons, collect enough observations across comparable route conditions and use a randomized rollout or a carefully specified causal evaluation. The included driver-clustered GEE improves inference for repeated observations within drivers, but any conclusion should still account for missing factors and the way work is assigned.

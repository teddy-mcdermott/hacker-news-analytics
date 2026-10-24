# Analysis

The analysis tools use uv dependency groups. From the repository root, install the web dependencies and start the web app with Docker:

```bash
uv sync --group web
uv run python -c "import Analysis.time_series.app"
docker compose up --build web
```

## Topic modelling

Topic modelling runs as a separate command with the `topics` dependency group. In Docker, the `topics` profile shares the cache and image volumes with the web service:

```bash
docker compose --profile topics run --rm topics python Analysis/topics/bertopic_analysis.py --days 30 --max-items 500
```

Or run locally:

```bash
uv sync --group topics
uv run python Analysis/topics/bertopic_analysis.py --days 30 --max-items 500
```

After computation finishes, the `/analyse_topics` endpoint returns the cached images and topic data. Its request parameters must match the values used for the computation.

## Other analysis scripts

Install the web group for time series and user analysis:

```bash
uv sync --group web
uv run python Analysis/users/top_users.py --limit 100
```

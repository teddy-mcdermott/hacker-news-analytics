FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM python:3.11-slim AS base
WORKDIR /app
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
COPY pyproject.toml uv.lock ./

FROM base AS scraper_image
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --group scraper
COPY ./helper ./helper
COPY ./Scraper ./Scraper
CMD ["python", "Scraper/dispatcher.py"]

FROM base AS analysis_image
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --group web
COPY ./helper ./helper
COPY ./Analysis ./Analysis
RUN mkdir -p /app/cache /app/Analysis/time_series/static/images
ENV PROJECT_ROOT=/app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "Analysis/time_series", "--timeout", "600", "app:app"]

FROM base AS topics_image
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --group topics
COPY ./helper ./helper
COPY ./Analysis ./Analysis
RUN mkdir -p /app/cache /app/Analysis/time_series/static/images
ENV PROJECT_ROOT=/app
CMD ["python", "Analysis/topics/bertopic_analysis.py"]

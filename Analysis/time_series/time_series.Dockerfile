FROM python:3.11-slim

WORKDIR /app

# 1. Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the necessary folders
COPY ./helper ./helper
COPY ./Analysis/time_series ./Analysis/time_series

# 3. Set Python path so it can find the 'helper' module
ENV PYTHONPATH=/app

# Tell Gunicorn to look inside the subfolder for 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "Analysis/time_series", "app:app"]
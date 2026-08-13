FROM python:3.12-slim

# Stockfish is the analysis oracle; the worker image is the only place it lives.
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*
ENV STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app
COPY pyproject.toml ./
COPY blunderless ./blunderless
RUN pip install --no-cache-dir .

# Placeholder until the Celery worker lands; proves the image builds and imports.
CMD ["python", "-c", "import blunderless; print('blunderless worker image OK', blunderless.__version__)"]

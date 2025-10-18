FROM python:3.11-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps for scientific python (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Pre-copy requirements for better caching
COPY scripts/train_two_tower/requirements.txt /tmp/requirements-train.txt
COPY scripts/prep_two_tower/requirements.txt /tmp/requirements-prep.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-train.txt && \
    pip install -r /tmp/requirements-prep.txt

# Default to an idle container; use `docker compose exec ml` to run commands
CMD ["sleep", "infinity"]

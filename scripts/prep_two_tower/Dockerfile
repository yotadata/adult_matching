FROM python:3.11-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Pre-copy requirements for layer caching
COPY scripts/prep_two_tower/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

CMD ["python", "scripts/prep_two_tower/prep_two_tower_dataset.py"]

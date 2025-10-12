FROM python:3.11-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY scripts/scrape_dmm_reviews/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

CMD ["python", "scripts/scrape_dmm_reviews/scrape_dmm_reviews.py"]

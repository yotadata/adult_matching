FROM python:3.11-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY scripts/update_user_embeddings/requirements.txt /tmp/requirements.txt

RUN printf '%s\n' \
  'deb https://deb.debian.org/debian bookworm main' \
  'deb https://deb.debian.org/debian bookworm-updates main' \
  'deb https://security.debian.org/debian-security bookworm-security main' \
  > /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

CMD ["python", "scripts/update_user_embeddings/main.py"]

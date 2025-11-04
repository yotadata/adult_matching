FROM python:3.11-bookworm

WORKDIR /workspace

COPY scripts/gen_video_embeddings/requirements.txt /tmp/requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

CMD ["python", "scripts/gen_video_embeddings/gen_video_embeddings.py"]

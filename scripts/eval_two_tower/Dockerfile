FROM python:3.11-bookworm

WORKDIR /workspace

COPY scripts/eval_two_tower/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

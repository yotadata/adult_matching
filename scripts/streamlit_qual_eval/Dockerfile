FROM python:3.11-bookworm

WORKDIR /workspace

COPY scripts/streamlit_qual_eval/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

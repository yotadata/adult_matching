FROM python:3.11-bookworm

WORKDIR /workspace

COPY scripts/gen_user_embeddings/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

CMD ["python", "scripts/gen_user_embeddings/gen_user_embeddings.py"]

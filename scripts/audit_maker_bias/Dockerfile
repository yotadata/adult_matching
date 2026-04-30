FROM python:3.11-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY audit_maker_bias.py .
ENTRYPOINT ["python", "audit_maker_bias.py"]

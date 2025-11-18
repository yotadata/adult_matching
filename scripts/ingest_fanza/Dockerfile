FROM node:20-bullseye

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_PATH=/opt/ingest/node_modules

WORKDIR /workspace

# Install runtime deps outside the bind-mounted workspace
RUN mkdir -p /opt/ingest && \
    cd /opt/ingest && \
    npm init -y && \
    npm install dotenv axios pg && \
    npm install -g tsx

# Default command runs the TS script
CMD ["tsx", "scripts/ingest_fanza/index.ts"]

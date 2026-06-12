FROM node:20-bullseye

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_PATH=/opt/ingest/node_modules

WORKDIR /workspace

# cheerio を追加（HTMLパーサー）
RUN mkdir -p /opt/ingest && \
    cd /opt/ingest && \
    npm init -y && \
    npm install @supabase/supabase-js dotenv axios ws cheerio && \
    npm install -g tsx

CMD ["tsx", "scripts/ingest_mgs/index.ts"]

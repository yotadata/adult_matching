FROM node:20-bullseye

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /workspace

# Minimal runtime deps + tsx for TS execution
RUN npm init -y && \
    npm install tsx @supabase/supabase-js dotenv axios

# Default command runs the TS script
CMD ["npx", "tsx", "scripts/ingest_fanza/index.ts"]


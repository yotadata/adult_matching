FROM ghcr.io/supabase/cli:latest

# Install bash + psql (Alpine base)
RUN apk add --no-cache bash postgresql15-client docker-cli

WORKDIR /workspace

CMD ["bash", "scripts/sync_remote_db/sync_remote_db.sh"]


FROM supabase/cli:latest

# Install bash + psql + docker cli (attempt Alpine first, then Debian/Ubuntu)
RUN (apk add --no-cache bash postgresql15-client docker-cli) || \
    (apt-get update && apt-get install -y bash postgresql-client docker.io && rm -rf /var/lib/apt/lists/*);

WORKDIR /workspace

CMD ["bash", "scripts/sync_remote_db/sync_remote_db.sh"]

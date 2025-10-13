# Frontend dev Dockerfile
# - Uses Node 20
# - Designed for hot reload with bind mounts

FROM node:20-bullseye

ENV NODE_ENV=development \
    NEXT_TELEMETRY_DISABLED=1 \
    WATCHPACK_POLLING=true \
    CHOKIDAR_USEPOLLING=true

WORKDIR /app

# Install dependencies separately for better caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy only the frontend sources for baseline (will be bind-mounted in dev)
COPY frontend/. ./

# Expose Next.js dev port
EXPOSE 3000

# Default command: run Next.js dev server
CMD ["npm", "run", "dev"]

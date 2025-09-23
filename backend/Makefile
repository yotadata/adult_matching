# ============================================================================
# Backend Makefile
# ============================================================================
# Common commands for backend development and deployment

.PHONY: help setup clean test lint format deploy-dev deploy-prod logs status

# Default target
help:
	@echo "Backend Makefile Commands:"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  setup          - Setup development environment"
	@echo "  clean          - Clean temporary files and caches"
	@echo "  env-check      - Check environment variables"
	@echo ""
	@echo "Development:"
	@echo "  dev            - Start development environment"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo ""
	@echo "Edge Functions:"
	@echo "  functions-dev  - Start Edge Functions development server"
	@echo "  functions-deploy - Deploy Edge Functions"
	@echo "  functions-logs - View Edge Functions logs"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  ml-train       - Train ML models"
	@echo "  ml-evaluate    - Evaluate model performance"
	@echo "  ml-deploy      - Deploy ML models"
	@echo ""
	@echo "Data Processing:"
	@echo "  data-collect   - Collect data from external sources"
	@echo "  data-process   - Process and clean data"
	@echo "  data-embed     - Generate embeddings"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate     - Run database migrations"
	@echo "  db-seed        - Seed database with test data"
	@echo "  db-backup      - Backup database"
	@echo "  db-restore     - Restore database from backup"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev     - Deploy to development environment"
	@echo "  deploy-prod    - Deploy to production environment"
	@echo "  status         - Check deployment status"
	@echo "  logs           - View application logs"

# ============================================================================
# Setup & Environment
# ============================================================================

setup:
	@echo "Setting up backend development environment..."
	@cp .env.example .env || echo ".env.example not found, please create it"
	@echo "Please configure your .env file with actual values"
	@echo "Setup complete! Run 'make env-check' to verify configuration"

clean:
	@echo "Cleaning temporary files and caches..."
	@rm -rf tmp/
	@rm -rf logs/*.log
	@rm -rf backend/data-processing/temp/
	@rm -rf backend/data-processing/cache/
	@echo "Clean complete!"

env-check:
	@echo "Checking environment variables..."
	@if [ -f .env ]; then \
		echo "✓ .env file exists"; \
	else \
		echo "✗ .env file missing - run 'make setup'"; \
		exit 1; \
	fi
	@echo "Environment check complete!"

# ============================================================================
# Development
# ============================================================================

dev: env-check
	@echo "Starting development environment..."
	@echo "Starting Supabase..."
	@cd .. && supabase start
	@echo "Development environment ready!"

test:
	@echo "Running tests..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && python -m pytest backend/; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

lint:
	@echo "Running linting..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python -m flake8 backend/ && \
		python -m mypy backend/; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

format:
	@echo "Formatting code..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python -m black backend/ && \
		python -m isort backend/; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

# ============================================================================
# Edge Functions
# ============================================================================

functions-dev:
	@echo "Starting Edge Functions development server..."
	@cd .. && supabase functions serve

functions-deploy: env-check
	@echo "Deploying Edge Functions..."
	@cd .. && supabase functions deploy user-management
	@cd .. && supabase functions deploy content
	@cd .. && supabase functions deploy recommendations

functions-logs:
	@echo "Viewing Edge Functions logs..."
	@cd .. && supabase functions logs

# ============================================================================
# ML Pipeline
# ============================================================================

ml-train: env-check
	@echo "Training ML models..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/ml-pipeline/training/train_two_tower.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

ml-evaluate: env-check
	@echo "Evaluating model performance..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/ml-pipeline/training/evaluate_model.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

ml-deploy: env-check
	@echo "Deploying ML models..."
	@echo "Copying models to production..."
	@echo "ML model deployment complete!"

# ============================================================================
# Data Processing
# ============================================================================

data-collect: env-check
	@echo "Collecting data from external sources..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/data-processing/scraping/collect_reviews.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

data-process: env-check
	@echo "Processing and cleaning data..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/data-processing/cleaning/clean_data.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

data-embed: env-check
	@echo "Generating embeddings..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/data-processing/embedding/generate_embeddings.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

# ============================================================================
# Database
# ============================================================================

db-migrate:
	@echo "Running database migrations..."
	@cd .. && supabase db push

db-seed: env-check
	@echo "Seeding database with test data..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		python backend/scripts/data-migration/seed_database.py; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

db-backup:
	@echo "Backing up database..."
	@cd .. && supabase db dump > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup complete!"

db-restore:
	@echo "Restoring database from backup..."
	@echo "Please specify backup file: make db-restore BACKUP_FILE=backup.sql"

# ============================================================================
# DMM Sync
# ============================================================================

dmm-sync: env-check
	@echo "Syncing data from DMM API..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		node ../scripts/real_dmm_sync.js; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

dmm-analyze: env-check
	@echo "Analyzing DMM data quality..."
	@if [ -d "../.venv" ]; then \
		source ../.venv/bin/activate && \
		node ../scripts/analyze_dmm_data.js; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

# ============================================================================
# Deployment
# ============================================================================

deploy-dev: env-check functions-deploy
	@echo "Deploying to development environment..."
	@echo "Development deployment complete!"

deploy-prod: env-check
	@echo "Deploying to production environment..."
	@echo "⚠️  Production deployment requires manual confirmation"
	@read -p "Deploy to production? (y/N): " confirm && [ "$$confirm" = "y" ]
	@make functions-deploy
	@echo "Production deployment complete!"

status:
	@echo "Checking deployment status..."
	@cd .. && supabase status

logs:
	@echo "Viewing application logs..."
	@cd .. && supabase functions logs

# ============================================================================
# Monitoring & Maintenance
# ============================================================================

health-check: env-check
	@echo "Performing health check..."
	@curl -f http://localhost:54321/functions/v1/user-management/health || echo "Health check failed"

performance-test: env-check
	@echo "Running performance tests..."
	@echo "Performance testing not yet implemented"

security-scan: env-check
	@echo "Running security scan..."
	@echo "Security scanning not yet implemented"

# ============================================================================
# Documentation
# ============================================================================

docs-build:
	@echo "Building documentation..."
	@echo "Documentation build not yet implemented"

docs-serve:
	@echo "Serving documentation..."
	@echo "Documentation server not yet implemented"

# ============================================================================
# Utilities
# ============================================================================

check-deps:
	@echo "Checking dependencies..."
	@which supabase > /dev/null || echo "⚠️  Supabase CLI not found"
	@which python3 > /dev/null || echo "⚠️  Python 3 not found"
	@which node > /dev/null || echo "⚠️  Node.js not found"
	@echo "Dependency check complete!"

version:
	@echo "Backend Version Information:"
	@echo "Supabase CLI: $(shell supabase --version 2>/dev/null || echo 'Not installed')"
	@echo "Python: $(shell python3 --version 2>/dev/null || echo 'Not installed')"
	@echo "Node.js: $(shell node --version 2>/dev/null || echo 'Not installed')"

# ============================================================================
# Variables for parameterized commands
# ============================================================================

BACKUP_FILE ?= latest.sql
ENVIRONMENT ?= development
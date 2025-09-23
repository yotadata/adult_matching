#!/bin/bash

# Two-Tower Model Training Example
# 
# Usage: ./train_example.sh
# 
# Make sure to set environment variables:
# - DATABASE_URL: PostgreSQL connection string
# - SUPABASE_URL: Supabase project URL
# - SUPABASE_SERVICE_ROLE_KEY: Supabase service role key

set -e

echo "Setting up environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable is not set"
    echo "Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'"
    exit 1
fi

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    source .env
fi

echo "Starting Two-Tower model training..."

# Run training
python train_two_tower_model.py \
    --db-url "$DATABASE_URL" \
    --embedding-dim 768 \
    --learning-rate 0.001 \
    --batch-size 64 \
    --epochs 20 \
    --output-dir "models/$(date +%Y%m%d_%H%M%S)" \
    --update-db

echo "Training completed successfully!"

# Optional: Upload models to Supabase Storage
if [ ! -z "$SUPABASE_URL" ] && [ ! -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "Uploading models to Supabase Storage..."
    # This would require a separate script to upload to Supabase Storage
    # python upload_models_to_supabase.py --model-dir "models/latest"
fi

echo "All done!"
# Project Structure

## Directory Organization

```
adult_matching/
├── frontend/                   # Next.js web application
│   ├── app/                   # Next.js App Router (pages and API routes)
│   ├── components/            # React components
│   ├── lib/                   # Utility libraries and Supabase client
│   └── public/                # Static assets
├── supabase/                  # Supabase backend configuration
│   ├── functions/             # Edge Functions (serverless API endpoints)
│   ├── migrations/            # Database schema migrations
│   └── config.toml            # Supabase project configuration
├── ml_pipeline/               # Python ML training and data processing
│   ├── training/              # Two-Tower model training scripts
│   ├── data_preprocessing/    # Data cleaning and preparation
│   ├── embedding_generation/  # Vector embedding creation
│   └── model_evaluation/      # Model testing and validation
├── data_processing/           # Data collection and ingestion
│   ├── scraping/              # Web scraping modules
│   ├── api_integration/       # External API handlers
│   └── data_validation/       # Data quality checks
├── scripts/                   # Utility and automation scripts
│   ├── dmm_*.js               # DMM/FANZA API integration scripts
│   ├── train_*.py             # ML model training scripts
│   └── data_*.js              # Data processing and analysis scripts
├── docs/                      # Project documentation
│   ├── specifications/        # Detailed technical specifications
│   └── *.md                   # Architecture and design documents
├── data/                      # Local data storage (gitignored)
│   ├── raw/                   # Raw scraped and API data
│   ├── processed/             # Cleaned and processed data
│   └── models/                # Trained ML model artifacts
└── .spec-workflow/            # Spec workflow steering documents
```

## Naming Conventions

### Files
- **React Components**: `PascalCase` (e.g., `SwipeCard.tsx`, `MobileVideoLayout.tsx`)
- **Pages/Routes**: `kebab-case` (e.g., `liked-videos`, `account-management`)
- **Utility Functions**: `camelCase` (e.g., `createSupabaseClient.ts`, `updateUserEmbedding.ts`)
- **Python Scripts**: `snake_case` (e.g., `train_two_tower_model.py`, `generate_embeddings.py`)
- **Configuration Files**: `lowercase` (e.g., `config.toml`, `pyproject.toml`)

### Code
- **React Components**: `PascalCase` (e.g., `SwipeCard`, `ActionButtons`)
- **Functions/Methods**: `camelCase` (e.g., `handleSwipe`, `updateUserEmbedding`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DMM_API_BASE_URL`, `EMBEDDING_DIMENSION`)
- **Variables**: `camelCase` (e.g., `userId`, `videoEmbedding`, `recommendationScore`)
- **Database Tables**: `snake_case` (e.g., `videos`, `user_embeddings`, `tag_groups`)

## Import Patterns

### Import Order
1. React and Next.js imports
2. Third-party libraries (Supabase, Framer Motion, etc.)
3. Internal utilities and components
4. Relative imports
5. CSS/style imports (Tailwind classes inline)

### Module Organization
```typescript
// External dependencies
import React from 'react'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { motion } from 'framer-motion'

// Internal modules
import { Database } from '@/lib/database.types'
import { updateUserEmbedding } from '@/lib/recommendations'

// Relative imports
import SwipeCard from './SwipeCard'
import ActionButtons from './ActionButtons'
```

## Code Structure Patterns

### React Component Organization
```typescript
// 1. Imports
import React, { useState, useEffect } from 'react'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'

// 2. Type definitions
interface ComponentProps {
  userId: string
  videoId: string
}

// 3. Main component implementation
export default function Component({ userId, videoId }: ComponentProps) {
  // 4. State and hooks
  const [loading, setLoading] = useState(false)
  const supabase = createClientComponentClient()
  
  // 5. Event handlers
  const handleAction = async () => { ... }
  
  // 6. Render method
  return (...)
}

// 7. Helper functions (if needed)
const helperFunction = () => { ... }
```

### Python Script Organization
```python
# 1. Standard library imports
import os
import json
from pathlib import Path

# 2. Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf

# 3. Local imports
from ml_pipeline.config import MODEL_CONFIG
from ml_pipeline.utils import load_data

# 4. Constants and configuration
EMBEDDING_DIM = 128
BATCH_SIZE = 1000

# 5. Main implementation
def train_model():
    """Main training function"""
    pass

# 6. Helper functions
def preprocess_data(data):
    """Data preprocessing helper"""
    pass

# 7. Script entry point
if __name__ == "__main__":
    train_model()
```

### File Organization Principles
- **One component per file**: Each React component in its own file
- **Functional grouping**: Related utilities grouped together
- **Clear public API**: Main exports at the top or bottom
- **Implementation details**: Helper functions at the end

## Code Organization Principles

1. **Single Responsibility**: Each component/function has one clear purpose
2. **Modularity**: Reusable components and utilities across the application
3. **Testability**: Structure allows easy unit and integration testing
4. **Consistency**: Follow established patterns throughout the codebase
5. **Type Safety**: Use TypeScript interfaces and Python type hints

## Module Boundaries

### Frontend Architecture
- **Pages**: Next.js App Router pages handle routing and server-side logic
- **Components**: Reusable UI components with clear props interfaces
- **Lib**: Shared utilities, Supabase client, and business logic
- **API Routes**: Server-side API endpoints for complex operations

### Backend Architecture
- **Edge Functions**: Serverless functions for recommendation logic
- **Database**: PostgreSQL with RLS policies for data access control
- **ML Pipeline**: Isolated Python environment for model training
- **Scripts**: Node.js utilities for data processing and API integration

### Data Flow Boundaries
- **Frontend ↔ Supabase**: Direct client connection with RLS security
- **Edge Functions ↔ Database**: Server-side database operations
- **ML Pipeline ↔ Database**: Batch processing and model updates
- **Scripts ↔ External APIs**: Data ingestion and synchronization

## Code Size Guidelines

- **React Components**: Maximum 200 lines per component file
- **Functions/Methods**: Maximum 50 lines per function
- **Python Scripts**: Maximum 500 lines per script file
- **Edge Functions**: Maximum 100 lines per function
- **Nesting Depth**: Maximum 4 levels of indentation

## Dashboard/Monitoring Structure

### Development Monitoring
```
Makefile commands:
├── make status          # Check overall project status
├── make data-collect    # Monitor data collection progress
├── make train           # Track ML model training
└── make setup          # Development environment setup
```

### Database Monitoring
```
supabase/
├── dashboard/           # Supabase web dashboard access
├── functions/          # Real-time function monitoring
└── migrations/         # Schema change tracking
```

### Separation of Concerns
- **Development Tools**: Make commands for pipeline monitoring
- **Production Monitoring**: Supabase dashboard for database/API monitoring
- **ML Pipeline**: Isolated Python environment with separate monitoring
- **Frontend**: Built-in Next.js development tools and hot reload

## Documentation Standards

### Code Documentation
- **React Components**: JSDoc comments for props and complex functions
- **Python Functions**: Docstrings following Google/NumPy style
- **Database Schema**: Comments in migration files
- **Edge Functions**: Clear parameter and return type documentation

### File Documentation
- **README Files**: Present in major directories (`frontend/`, `ml_pipeline/`)
- **Specification Files**: Detailed docs in `docs/specifications/`
- **API Documentation**: OpenAPI spec for Edge Functions
- **Setup Instructions**: Clear installation and development guides in root README

### Comment Guidelines
- **Complex Business Logic**: Explain why, not what
- **ML Algorithms**: Document model architecture and training parameters
- **Database Queries**: Explain complex joins and performance considerations
- **External API Integration**: Document rate limits and error handling
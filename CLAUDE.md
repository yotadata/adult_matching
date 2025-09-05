# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an adult video matching application built with Next.js and Supabase. The application provides a Tinder-like swipe interface for users to discover and like adult videos, with AI-powered recommendations using vector embeddings.

## Development Commands

**Frontend (run from `/frontend` directory):**
- `npm run dev` - Start Next.js development server on localhost:3000
- `npm run build` - Build production Next.js app  
- `npm run start` - Start production Next.js server
- `npm run lint` - Run ESLint for code linting

**Database:**
- Migration files located in `/db/migrations/`
- Initial schema: `/db/migrations/20250818000000000_initial.sql`

**Data Processing & ML Pipeline:**
- `make setup` - Initial project setup and environment preparation
- `make data-collect` - Collect review data using web scraping (requires cookie setup)
- `make data-clean` - Clean and preprocess collected data  
- `make data-embed` - Generate embedding vectors from processed text
- `make train` - Train Two-Tower recommendation model
- `make train-full` - Full pipeline: data collection → cleaning → embedding → training
- `make status` - Check project status and data availability
- Python 3.12.3 environment located in `.venv/` (managed by uv v0.8.11)

**Data Ingestion:**
- `scripts/fanza_ingest.ts` - Script for ingesting video data from FANZA
- `data_processing/scraping/` - Web scraping modules for initial data collection
- `ml_pipeline/training/` - Machine learning model training pipeline

### Python Environment Details
**Configuration Files:**
- `/pyproject.toml` - Main project configuration with dependencies
- `/scripts/requirements.txt` - Detailed requirements with additional ML packages
- `/uv.lock` - Dependency lock file for reproducible builds

**Key Dependencies:**
- **ML Frameworks:** tensorflow>=2.16.0, scikit-learn>=1.3.0, torch>=2.0.0
- **Data Processing:** numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0  
- **Database:** psycopg2-binary>=2.9.0 (PostgreSQL connectivity)
- **Model Deployment:** onnx, onnxruntime, tf2onnx, tensorflowjs
- **Development:** jupyter>=1.0.0, ipython>=8.0.0
- **Visualization:** matplotlib>=3.7.0, seaborn>=0.12.0, plotly>=5.15.0
- **Testing:** pytest>=7.4.0, pytest-cov>=4.1.0
- **Utilities:** python-dotenv, tqdm, pyyaml

**Virtual Environment:**
- Location: `.venv/` directory
- Python Version: 3.12.3
- System Site Packages: Disabled
- Editable Installation: Project installed as `adult-matching-two-tower`

**Usage:**
- Environment is automatically activated when using `uv run`
- For direct activation: `source .venv/bin/activate`
- Package management: `uv add <package>`, `uv remove <package>`

## Architecture

### Frontend Structure
- **Next.js App Router** (App Directory structure)
- **Components:**
  - `SwipeCard.tsx` - Main swipeable video card component with drag animations
  - `MobileVideoLayout.tsx` - Mobile-optimized video layout
  - `ActionButtons.tsx` - Like/skip action buttons for desktop
  - `Header.tsx` - Application header
  - `AuthModal.tsx` - Authentication modal with login/signup forms
- **Pages:**
  - `/` - Main swipe interface
  - `/liked-videos` - User's liked videos
  - `/analysis-results` - Analysis results page
  - `/account-management` - Account management
- **API Routes:** Authentication endpoints in `/api/auth/`

### Database Schema (PostgreSQL + Supabase)
- **videos** - Video metadata with vector embeddings support
- **tags/tag_groups** - Flexible tagging system
- **performers** - Actor/performer information
- **likes** - User likes with RLS (Row Level Security)
- **user_embeddings** - User preference vectors for recommendations
- **video_embeddings** - Video content vectors for matching

### Key Technologies
- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS
- **Animation:** Framer Motion for swipe gestures and transitions
- **Database:** Supabase (PostgreSQL) with vector extension for embeddings
- **Authentication:** Supabase Auth
- **State Management:** React hooks (no external state library)

### API Architecture
The application uses Supabase Edge Functions for backend logic:
- `/feed_explore` - Initial diverse video feed for new users
- `/recommendations` - Personalized recommendations using Two-Tower model
- `/update_user_embedding` - Updates user preference vectors
- `/likes` - User's liked videos
- `/delete_account` - Account deletion

### Mobile vs Desktop
The app has responsive design with different layouts:
- **Desktop:** Card-based swipe interface with action buttons
- **Mobile:** Full-screen video layout optimized for mobile gestures

## Data Specifications & Documentation

### 📚 Complete Specification Library
**Location**: `docs/specifications/data/`

#### **Core Index**
- **[docs/specifications/data/README.md](docs/specifications/data/README.md)** - Main data specification index with data flow overview

#### **Database Specifications** (`docs/specifications/data/database/`)
- **[schema.md](docs/specifications/data/database/schema.md)** - Complete PostgreSQL schema, RLS policies, indexes
- **[migrations.md](docs/specifications/data/database/migrations.md)** - Database migration management guidelines
- **[rls_policies.md](docs/specifications/data/database/rls_policies.md)** - Row Level Security detailed policies

#### **Scraped Data Specifications** (`docs/specifications/data/scraped/`)
- **[reviews_format.md](docs/specifications/data/scraped/reviews_format.md)** - Review data format and structure
- **[batch_collection_process.md](docs/specifications/data/scraped/batch_collection_process.md)** - Large-scale data collection process
- **[data_quality_standards.md](docs/specifications/data/scraped/data_quality_standards.md)** - Data quality validation and standards

#### **API Integration Specifications** (`docs/specifications/data/api/`)
- **[dmm_fanza_integration.md](docs/specifications/data/api/dmm_fanza_integration.md)** - DMM/FANZA API integration specs
- **[supabase_functions.md](docs/specifications/data/api/supabase_functions.md)** - Supabase Edge Functions documentation
- **[external_apis.md](docs/specifications/data/api/external_apis.md)** - External API integration patterns

#### **Machine Learning Specifications** (`docs/specifications/data/ml/`)
- **[training_data_specs.md](docs/specifications/data/ml/training_data_specs.md)** - Complete ML training data specifications
- **[pseudo_user_generation.md](docs/specifications/data/ml/pseudo_user_generation.md)** - Pseudo-user generation methodology
- **[embeddings_management.md](docs/specifications/data/ml/embeddings_management.md)** - Vector embeddings management
- **[model_artifacts.md](docs/specifications/data/ml/model_artifacts.md)** - ML model file management

#### **Data Format Specifications** (`docs/specifications/data/formats/`)
- **[json_schemas.md](docs/specifications/data/formats/json_schemas.md)** - JSON file format definitions
- **[file_system_organization.md](docs/specifications/data/formats/file_system_organization.md)** - File system structure
- **[data_flow_diagrams.md](docs/specifications/data/formats/data_flow_diagrams.md)** - Data flow architecture diagrams

### 🔧 Development Requirements - MANDATORY SPECIFICATION REFERENCE

**CRITICAL: When working on any development task, you MUST reference the appropriate specifications:**

#### **Database Work** → Reference:
- `docs/specifications/data/database/schema.md` for table structures
- `docs/specifications/data/database/rls_policies.md` for security policies
- `docs/specifications/data/database/migrations.md` for schema changes

#### **Data Processing** → Reference:
- `docs/specifications/data/scraped/reviews_format.md` for review data structure
- `docs/specifications/data/scraped/batch_collection_process.md` for collection workflows
- `docs/specifications/data/scraped/data_quality_standards.md` for validation rules

#### **API Development** → Reference:
- `docs/specifications/data/api/dmm_fanza_integration.md` for external API work
- `docs/specifications/data/api/supabase_functions.md` for Edge Function development
- `docs/specifications/data/api/external_apis.md` for API integration patterns

#### **ML Pipeline Work** → Reference:
- `docs/specifications/data/ml/training_data_specs.md` for data format requirements
- `docs/specifications/data/ml/pseudo_user_generation.md` for user data conversion
- `docs/specifications/data/ml/embeddings_management.md` for vector operations

#### **File Format Work** → Reference:
- `docs/specifications/data/formats/json_schemas.md` for JSON structure validation
- `docs/specifications/data/formats/file_system_organization.md` for file placement
- `docs/specifications/data/formats/data_flow_diagrams.md` for architecture understanding

### ⚠️ Specification Compliance Rules

1. **BEFORE** starting any data-related task, read the relevant specification documents
2. **DURING** development, validate all changes against specification requirements  
3. **AFTER** implementation, update specifications if data structures or processes change
4. **NEVER** implement data changes without consulting the appropriate specification first
5. **ALWAYS** maintain consistency between code implementation and specification documentation

### 🎯 **CRITICAL DATA POLICY - API-First Video Data**

**MANDATORY VIDEO DATA POLICY:**
- **Video Data Source**: ONLY use API-retrieved data from DMM/FANZA APIs stored in PostgreSQL `videos` table
- **Scraped Review Data**: Use ONLY for Content ID linking and pseudo-user generation
- **ML Training**: Combine API video features + review-derived pseudo-user interactions
- **NO** use of scraped video metadata for training or recommendations

**Implementation Requirements:**
- Item features MUST come from `videos` table (API data)
- Content ID matching between reviews and API videos is required
- All video metadata for ML must be API-sourced
- Review data serves as behavioral signal only, not content metadata

## Important Notes
- Uses vector similarity search for video recommendations
- Implements Row Level Security (RLS) for user data protection  
- Supports both online and batch embedding updates
- Video data is ingested from external sources (FANZA)
- Authentication managed through Supabase Auth
- Environment variables required: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
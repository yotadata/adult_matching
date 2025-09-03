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

**Data Ingestion:**
- `scripts/fanza_ingest.ts` - Script for ingesting video data from FANZA

**Python Environment (Two-Tower Model):**
- `uv sync` - Install Python dependencies for Two-Tower model training
- `uv run python scripts/train_two_tower_model.py` - Train Two-Tower recommendation model
- `uv run python scripts/batch_embedding_update.py` - Update embeddings in batch
- `bash scripts/run_batch_update.sh` - Run batch embedding update script
- Python 3.12.3 environment located in `.venv/` (managed by uv v0.8.11)

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

## Important Notes
- Uses vector similarity search for video recommendations
- Implements Row Level Security (RLS) for user data protection  
- Supports both online and batch embedding updates
- Video data is ingested from external sources (FANZA)
- Authentication managed through Supabase Auth
- Environment variables required: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`/
# Technology Stack

## Project Type
Full-stack web application with machine learning pipeline - an adult video matching platform with AI-powered recommendations, featuring both web interface and data processing/ML training components.

## Core Technologies

### Primary Language(s)
- **TypeScript**: Frontend and Supabase Edge Functions (Next.js App Router)
- **JavaScript**: Node.js scripts for data processing and API integration
- **Python 3.12.3**: Machine learning pipeline, data processing, and model training
- **SQL**: PostgreSQL database schema and queries

### Key Dependencies/Libraries

**Frontend Stack:**
- **Next.js 15**: React framework with App Router for full-stack development
- **React 19**: UI library with hooks-based state management
- **Tailwind CSS**: Utility-first styling framework
- **Framer Motion**: Animation library for swipe gestures and transitions
- **Supabase Client**: Real-time database and authentication

**Backend Stack:**
- **Supabase**: PostgreSQL database with Edge Functions runtime
- **Deno**: Edge Functions runtime environment
- **Node.js**: Data processing scripts and API integration

**Machine Learning Stack:**
- **TensorFlow 2.16+**: Deep learning framework for Two-Tower model
- **scikit-learn 1.3+**: Classical ML algorithms and data preprocessing
- **PyTorch 2.0+**: Alternative deep learning framework
- **pandas 2.0+**: Data manipulation and analysis
- **NumPy 1.24+**: Numerical computing foundation

### Application Architecture
- **Frontend**: Next.js App Router with React components and Tailwind styling
- **Backend**: Supabase Edge Functions for API endpoints and business logic
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **ML Pipeline**: Python-based training pipeline with model artifacts export
- **Authentication**: Supabase Auth with Row Level Security (RLS) policies
- **Real-time Updates**: Supabase real-time subscriptions for live data

### Data Storage
- **Primary Storage**: PostgreSQL (Supabase) with vector extension for embeddings
- **Caching**: Supabase built-in connection pooling and query caching
- **Data Formats**: JSON for API responses, binary vectors for ML embeddings
- **File Storage**: Local file system for ML model artifacts and training data

### External Integrations
- **DMM/FANZA API**: Video metadata and content information retrieval
- **APIs**: RESTful integration with rate limiting (1 call/second)
- **Authentication**: Supabase Auth with social login options
- **Protocols**: HTTP/REST for API communication, WebSocket for real-time updates

### Monitoring & Dashboard Technologies
- **Dashboard**: Supabase built-in dashboard for database monitoring
- **Real-time Communication**: Supabase WebSocket subscriptions
- **Development Monitoring**: Make commands for pipeline status tracking
- **State Management**: React hooks and Supabase real-time state synchronization

## Development Environment

### Build & Development Tools
- **Frontend Build**: Next.js with automatic TypeScript compilation
- **Python Environment**: uv (0.8.11) for dependency management and virtual environments
- **Package Management**: npm for Node.js/frontend, uv for Python packages
- **Development Workflow**: 
  - `npm run dev` for hot-reload frontend development
  - `make` commands for ML pipeline development
  - `supabase functions serve` for Edge Functions development

### Code Quality Tools
- **Static Analysis**: TypeScript compiler, ESLint for JavaScript/TypeScript
- **Formatting**: Prettier for code formatting
- **Testing Framework**: Next.js built-in testing utilities
- **Python Quality**: pytest for unit testing, type hints for static analysis

### Version Control & Collaboration
- **VCS**: Git with feature branch workflow
- **Branching Strategy**: Feature branches with dev and main branches
- **Code Review Process**: GitHub-based pull request reviews

### Dashboard Development
- **Live Reload**: Next.js hot module replacement
- **Port Management**: Next.js on :3000, Supabase local on :54321
- **Multi-Instance Support**: Separate development and production Supabase projects

## Deployment & Distribution

- **Target Platforms**: Web browsers (responsive mobile/desktop)
- **Frontend Deployment**: Vercel or similar Next.js hosting platforms
- **Backend Deployment**: Supabase hosted PostgreSQL and Edge Functions
- **ML Pipeline**: Local development environment with cloud model deployment
- **Installation**: Web application - no user installation required

## Technical Requirements & Constraints

### Performance Requirements
- **Recommendation Response Time**: <500ms for AI-powered video suggestions
- **Swipe Interface Latency**: <50ms gesture response for smooth UX
- **Database Query Performance**: Vector similarity searches optimized with indexes
- **ML Model Inference**: Real-time embedding generation and similarity calculation

### Compatibility Requirements
- **Browser Support**: Modern browsers with ES2020+ support
- **Mobile Responsiveness**: Touch-optimized interface for iOS/Android browsers
- **Database Compatibility**: PostgreSQL 14+ with pgvector extension
- **Python Version**: 3.12.3 for ML pipeline compatibility

### Security & Compliance
- **Authentication**: Supabase Auth with secure session management
- **Data Protection**: Row Level Security (RLS) for user data isolation
- **API Security**: Rate limiting and input validation for external APIs
- **Adult Content Compliance**: Age verification and appropriate content handling

### Scalability & Reliability
- **Database Scaling**: Supabase managed PostgreSQL with automatic scaling
- **Vector Search Performance**: Optimized indexes for large video databases (200k+)
- **ML Pipeline Scalability**: Batch processing for large-scale embedding generation
- **Edge Functions**: Serverless scaling for API endpoints

## Technical Decisions & Rationale

### Decision Log
1. **Next.js App Router**: Chosen for full-stack TypeScript development with server-side rendering and optimized performance
2. **Supabase over Custom Backend**: Reduces development complexity while providing PostgreSQL, real-time features, and authentication
3. **Two-Tower ML Architecture**: Proven recommendation system design for user-item matching at scale
4. **Vector Embeddings for Recommendations**: Enables semantic similarity search and personalized content discovery
5. **DMM API Integration**: Ensures high-quality, complete video metadata over scraped data
6. **Python ML Pipeline**: Leverages mature ML ecosystem while maintaining separation from web application

## Known Limitations

- **Edge Function Limitations**: Some complex operations require Node.js scripts as fallback
- **Vector Search Performance**: Large-scale similarity searches may require optimization for 200k+ videos
- **ML Model Updates**: Manual model retraining process - could benefit from automated pipeline
- **API Rate Limits**: DMM API rate limiting requires careful batch processing for large syncs
- **Adult Content Restrictions**: Platform deployment options may be limited due to content nature
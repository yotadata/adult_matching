# Technology Stack

## Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **UI Framework**: React 19
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion (スワイプジェスチャーとトランジション)
- **Icon**: Lucide React
- **UI Components**: Headless UI
- **Form Handling**: React Hook Form
- **Notifications**: React Hot Toast

## Backend & Database
- **BaaS**: Supabase
- **Database**: PostgreSQL with Vector extension
- **Authentication**: Supabase Auth
- **API**: Supabase Edge Functions (Deno/TypeScript)
- **Vector Search**: pgvector extension for embeddings

## AI/ML Stack
- **Training Environment**: Python, TensorFlow, scikit-learn
- **Model Format**: TensorFlow.js / ONNX.js for browser deployment
- **Embeddings**: 768-dimensional vectors
- **Database**: PostgreSQL with vector extension
- **Feature Engineering**: pandas, numpy

## Development Tools
- **Linting**: ESLint with Next.js config
- **Type Checking**: TypeScript strict mode
- **Package Manager**: npm
- **Version Control**: Git
- **Environment**: Linux development environment

## Infrastructure
- **Hosting**: Supabase (Edge Functions)
- **CDN**: Next.js built-in optimization
- **Storage**: Supabase Storage (for model files)
- **Security**: Row Level Security (RLS) for data protection

## Key Dependencies
```json
{
  "next": "15.4.6",
  "react": "19.1.0", 
  "@supabase/supabase-js": "^2.55.0",
  "framer-motion": "^12.23.12",
  "typescript": "^5"
}
```
/**
 * Edge Functions共通型定義
 */

// ============================================================================
// 基本型定義
// ============================================================================

export interface User {
  id: string;
  email?: string;
  created_at?: string;
  updated_at?: string;
}

export interface DatabaseError {
  code: string;
  message: string;
  details?: any;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  code?: string;
  message?: string;
}

export interface PaginationInfo {
  limit: number;
  offset: number;
  total_count?: number;
  has_more?: boolean;
}

// ============================================================================
// 認証関連型定義
// ============================================================================

export interface AuthResult {
  user_id: string;
  email?: string;
  authenticated: boolean;
  error?: string;
}

export interface AuthHeaders {
  'Access-Control-Allow-Origin': string;
  'Access-Control-Allow-Headers': string;
  'Access-Control-Allow-Methods': string;
}

// ============================================================================
// 動画コンテンツ型定義
// ============================================================================

// 旧Video型 - DEPRECATED: genreカラムは本番環境に存在しない
// 後方互換性のため一時的に保持、段階的にVideoWithTagsに移行
export interface Video {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  genre: string; // DEPRECATED: 使用禁止
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  duration_seconds?: number;
  created_at?: string;
  updated_at?: string;
}

// 新しいtags-basedビデオ型 - 推奨使用
export interface VideoWithTags {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  duration_seconds?: number;
  // Tags-basedプロパティ
  genre: string;        // tags JOINから取得、Unknown保証
  all_tags: string[];   // 動画に関連する全タグ
  created_at?: string;
  updated_at?: string;
}

export interface VideoWithRelations extends VideoWithTags {
  performers: string[];
  tags: string[]; // all_tagsのエイリアス（後方互換性）
  similarity_score?: number;
  recommendation_reason?: string;
}

export interface VideoEmbedding {
  video_id: string;
  embedding: number[];
  updated_at: string;
}

// ============================================================================
// ユーザー関連型定義
// ============================================================================

export interface UserProfile {
  user_id: string;
  display_name?: string;
  avatar_url?: string;
  preferences?: Record<string, any>;
  is_admin?: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface UserEmbedding {
  user_id: string;
  embedding: number[];
  updated_at: string;
}

export interface UserLike {
  user_id: string;
  video_id: string;
  purchased: boolean;
  created_at: string;
}

export interface UserBehavior {
  user_id: string;
  action: 'like' | 'view' | 'purchase' | 'search';
  video_id?: string;
  metadata?: Record<string, any>;
  created_at: string;
}

// ============================================================================
// 推薦システム型定義
// ============================================================================

export interface RecommendationRequest {
  user_id?: string;
  limit?: number;
  offset?: number;
  algorithm?: 'enhanced' | 'basic' | 'two_tower' | 'collaborative' | 'content_based';
  diversity_weight?: number;
  include_reasons?: boolean;
  exclude_liked?: boolean;
  min_similarity_threshold?: number;
  filters?: ContentFilters;
  // 後方互換性
  max_results?: number;
  include_explanations?: boolean;
}

export interface RecommendationResponse {
  videos: VideoWithRelations[];
  total_count: number;
  algorithm: string;
  diversity_metrics?: DiversityMetrics;
  pagination?: PaginationInfo;
  performance_metrics?: PerformanceMetrics;
  // Tags-based metadata
  genre_distribution?: Record<string, number>;
  tag_distribution?: Record<string, number>;
}

export interface ContentFilters {
  genres?: string[];
  makers?: string[];
  performers?: string[];
  tags?: string[];
  price_min?: number;
  price_max?: number;
  duration_min?: number;
  duration_max?: number;
  content_ids?: string[];
  created_after?: string;
  created_before?: string;
}

// ============================================================================
// メトリクス型定義
// ============================================================================

export interface DiversityMetrics {
  genre_diversity: number;
  maker_diversity: number;
  performer_diversity?: number;
  tag_diversity?: number;
  price_range: {
    min: number;
    max: number;
    avg: number;
    std?: number;
  };
  content_freshness: number;
}

export interface PerformanceMetrics {
  operation_name: string;
  duration_ms: number;
  memory_usage?: number;
  database_queries?: number;
  cache_hits?: number;
  cache_misses?: number;
  timestamp: string;
}

export interface QualityMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  diversity_score?: number;
  novelty_score?: number;
  coverage?: number;
}

// ============================================================================
// バッチ処理型定義
// ============================================================================

export interface BatchJob {
  id: string;
  type: 'embedding_update' | 'recommendation_training' | 'data_cleanup';
  status: 'pending' | 'running' | 'completed' | 'failed';
  params: Record<string, any>;
  progress?: {
    current: number;
    total: number;
    percentage: number;
  };
  result?: Record<string, any>;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface BatchProgress {
  phase: string;
  batch_size: number;
  completed_items: number;
  total_items?: number;
  estimated_remaining?: number;
  next_phase?: string;
}

// ============================================================================
// 検索関連型定義
// ============================================================================

export interface SearchRequest {
  query?: string;
  filters?: ContentFilters;
  pagination?: PaginationInfo;
  sort_by?: 'relevance' | 'created_at' | 'price' | 'title' | 'genre' | 'popularity';
  sort_order?: 'asc' | 'desc';
  search_type?: 'text' | 'semantic' | 'hybrid';
}

export interface SearchResponse {
  videos: VideoWithRelations[];
  total_count: number;
  search_type: string;
  query?: string;
  filters?: ContentFilters;
  metrics?: DiversityMetrics;
  pagination?: PaginationInfo;
  performance?: PerformanceMetrics;
  // Tags-based search metadata
  matched_genres?: string[];
  matched_tags?: string[];
}

// ============================================================================
// ログ関連型定義
// ============================================================================

export interface LogLevel {
  DEBUG: 'debug';
  INFO: 'info';
  WARN: 'warn';
  ERROR: 'error';
}

export interface LogEntry {
  level: keyof LogLevel;
  message: string;
  timestamp: string;
  user_id?: string;
  function_name?: string;
  operation_id?: string;
  metadata?: Record<string, any>;
  error?: Error | string;
}

// ============================================================================
// 設定関連型定義
// ============================================================================

export interface EdgeFunctionConfig {
  cors_origins: string[];
  rate_limits: {
    authenticated: number;
    anonymous: number;
    admin: number;
  };
  cache_ttl: {
    user_embeddings: number;
    video_recommendations: number;
    search_results: number;
  };
  ml_model_params: {
    embedding_dimension: number;
    similarity_threshold: number;
    diversity_weight: number;
  };
}

// ============================================================================
// イベント関連型定義
// ============================================================================

export interface UserEvent {
  event_type: 'page_view' | 'video_like' | 'video_unlike' | 'search' | 'recommendation_click';
  user_id?: string;
  session_id?: string;
  video_id?: string;
  search_query?: string;
  recommendation_context?: {
    algorithm: string;
    position: number;
    similarity_score?: number;
  };
  metadata?: Record<string, any>;
  timestamp: string;
}

// ============================================================================
// エラー関連型定義
// ============================================================================

export interface ErrorInfo {
  code: string;
  message: string;
  details?: any;
  function_name?: string;
  user_id?: string;
  timestamp?: string;
  stack_trace?: string;
}

export type ErrorCode = 
  | 'UNAUTHORIZED'
  | 'FORBIDDEN'
  | 'NOT_FOUND'
  | 'BAD_REQUEST'
  | 'INTERNAL_ERROR'
  | 'DATABASE_ERROR'
  | 'EXTERNAL_API_ERROR'
  | 'VALIDATION_ERROR'
  | 'RATE_LIMIT_EXCEEDED'
  | 'INSUFFICIENT_DATA'
  | 'MODEL_ERROR'
  | 'EMBEDDING_ERROR';
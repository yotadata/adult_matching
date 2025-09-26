/**
 * 統一コンテンツ処理ユーティリティ
 * Unified content processing utilities for Edge Functions
 */

import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

// Inline type definitions to avoid import issues
interface DatabaseResult<T> {
  success: boolean;
  data?: T;
  error?: string;
}

interface PaginationParams {
  limit?: number;
  offset?: number;
}

interface FilterParams {
  content_ids?: string[];
}

interface VideoWithRelations {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  sample_video_url?: string;
  maker: string;
  genre: string;
  price: number;
  duration_seconds?: number;
  image_urls: string[];
  performers: string[];
  tags: string[];
  created_at?: string;
  similarity_score?: number;
}

interface ContentFilters {
  genres?: string[];
  makers?: string[];
  price_min?: number;
  price_max?: number;
  duration_min?: number;
  duration_max?: number;
  created_after?: string;
  created_before?: string;
  content_ids?: string[];
}

interface DiversityMetrics {
  genre_diversity: number;
  maker_diversity: number;
  price_range: { min: number; max: number; avg: number };
  content_freshness: number;
}

interface PerformanceMetrics {
  operation_name: string;
  duration_ms: number;
  memory_usage: number;
  timestamp: string;
}

// Utility functions
function createSuccessResult<T>(data: T): DatabaseResult<T> {
  return { success: true, data };
}

function handleDatabaseError(error: any): DatabaseResult<any> {
  console.error('Database error:', error);
  return { success: false, error: error.message || 'Database operation failed' };
}

function applyPagination(query: any, pagination: PaginationParams): any {
  const { limit = 20, offset = 0 } = pagination;
  return query.range(offset, offset + limit - 1);
}

// ============================================================================
// コンテンツ取得と処理
// ============================================================================

/**
 * 動画詳細情報を取得（リレーション含む）
 * Get video details with relations
 */
export async function getVideoContent(
  supabase: SupabaseClient,
  videoIds: string[]
): Promise<VideoWithRelations[]> {
  if (videoIds.length === 0) return [];

  const { data: videos, error } = await supabase
    .from('videos')
    .select(`
      id,
      title,
      description,
      thumbnail_url,
      preview_video_url,
      sample_video_url,
      maker,
      price,
      duration_seconds,
      image_urls,
      created_at,
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
    `)
    .in('id', videoIds);

  if (error) {
    console.error('Error fetching video content:', error);
    return [];
  }

  return videos?.map((video: any) => {
    const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
    const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

    return {
      id: video.id,
      title: video.title,
      description: video.description || '',
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      sample_video_url: video.sample_video_url,
      maker: video.maker,
      genre: genreTag, // tagsから動的に生成されたgenre
      price: video.price || 0,
      duration_seconds: video.duration_seconds,
      image_urls: video.image_urls || [],
      performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: tags,
      created_at: video.created_at
    };
  }) || [];
}

/**
 * ベクター類似度による動画検索
 * Get videos by vector similarity
 */
export async function getVideosBySimilarity(
  supabase: SupabaseClient,
  userEmbedding: number[],
  limit: number = 20,
  excludeIds: string[] = [],
  similarityThreshold: number = 0.1
): Promise<VideoWithRelations[]> {
  try {
    // Use the RPC function for vector similarity search
    let rpcQuery = supabase.rpc('match_videos_cosine', {
      query_embedding: userEmbedding,
      match_threshold: similarityThreshold,
      match_count: limit
    });

    const { data: matches, error } = await rpcQuery;

    if (error) {
      console.error('Error in vector similarity search:', error);
      throw new Error('Failed to perform similarity search');
    }

    if (!matches || matches.length === 0) {
      return [];
    }

    // Filter out excluded IDs
    const filteredMatches = matches.filter((match: any) =>
      !excludeIds.includes(match.id)
    );

    // Get video details
    const videoIds = filteredMatches.map((match: any) => match.id);
    const videos = await getVideoContent(supabase, videoIds);

    // Add similarity scores
    return videos.map(video => {
      const match = filteredMatches.find((m: any) => m.id === video.id);
      return {
        ...video,
        similarity_score: match?.similarity || 0
      };
    });
  } catch (error) {
    console.error('Error in getVideosBySimilarity:', error);
    return [];
  }
}

/**
 * フィルタ条件による動画検索
 * Search videos with filters
 */
export async function searchVideosWithFilters(
  supabase: SupabaseClient,
  filters: ContentFilters,
  pagination: PaginationParams = {}
): Promise<DatabaseResult<{ videos: VideoWithRelations[]; total_count: number }>> {
  try {
    let query = supabase
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        sample_video_url,
        maker,
        price,
        duration_seconds,
        image_urls,
        created_at,
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `, { count: 'exact' });

    // Apply content filters - tags-based genre filtering
    if (filters.genres && filters.genres.length > 0) {
      query = query.in('id',
        supabase
          .from('video_tags')
          .select('video_id')
          .in('tag_id',
            supabase
              .from('tags')
              .select('id')
              .in('name', filters.genres)
          )
      );
    }

    if (filters.makers && filters.makers.length > 0) {
      query = query.in('maker', filters.makers);
    }

    if (filters.price_min !== undefined) {
      query = query.gte('price', filters.price_min);
    }

    if (filters.price_max !== undefined) {
      query = query.lte('price', filters.price_max);
    }

    if (filters.duration_min !== undefined) {
      query = query.gte('duration_seconds', filters.duration_min);
    }

    if (filters.duration_max !== undefined) {
      query = query.lte('duration_seconds', filters.duration_max);
    }

    if (filters.created_after) {
      query = query.gte('created_at', filters.created_after);
    }

    if (filters.created_before) {
      query = query.lte('created_at', filters.created_before);
    }

    if (filters.content_ids && filters.content_ids.length > 0) {
      query = query.not('id', 'in', filters.content_ids);
    }

    // Apply pagination
    query = applyPagination(query, pagination);

    // Execute query
    const { data: videos, error, count } = await query;

    if (error) {
      return handleDatabaseError(error);
    }

    const formattedVideos: VideoWithRelations[] = videos?.map((video: any) => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

      return {
        id: video.id,
        title: video.title,
        description: video.description || '',
        thumbnail_url: video.thumbnail_url,
        preview_video_url: video.preview_video_url,
        sample_video_url: video.sample_video_url,
        maker: video.maker,
        genre: genreTag, // tagsから動的に生成されたgenre
        price: video.price || 0,
        duration_seconds: video.duration_seconds,
        image_urls: video.image_urls || [],
        performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        tags: tags,
        created_at: video.created_at
      };
    }) || [];

    return createSuccessResult({
      videos: formattedVideos,
      total_count: count || 0
    });

  } catch (error) {
    return handleDatabaseError(error);
  }
}

/**
 * コンテンツの多様性を向上させる
 * Diversify content selection
 */
export function diversifyContent(
  videos: VideoWithRelations[],
  diversityWeight: number = 0.3
): VideoWithRelations[] {
  if (videos.length <= 1) return videos;

  // Group by genre and maker for diversity
  const genreGroups = new Map<string, VideoWithRelations[]>();
  const makerGroups = new Map<string, VideoWithRelations[]>();

  videos.forEach(video => {
    if (!genreGroups.has(video.genre)) {
      genreGroups.set(video.genre, []);
    }
    genreGroups.get(video.genre)!.push(video);

    if (!makerGroups.has(video.maker)) {
      makerGroups.set(video.maker, []);
    }
    makerGroups.get(video.maker)!.push(video);
  });

  // Interleave videos from different groups
  const diversified: VideoWithRelations[] = [];
  const remaining = [...videos];

  while (remaining.length > 0 && diversified.length < videos.length) {
    // Try to pick from different genre
    const lastGenre = diversified.length > 0 ? diversified[diversified.length - 1].genre : null;
    let nextVideo = remaining.find(v => v.genre !== lastGenre);

    if (!nextVideo) {
      // If no different genre available, pick by different maker
      const lastMaker = diversified.length > 0 ? diversified[diversified.length - 1].maker : null;
      nextVideo = remaining.find(v => v.maker !== lastMaker);
    }

    if (!nextVideo) {
      // If no diversity possible, just pick the first remaining
      nextVideo = remaining[0];
    }

    diversified.push(nextVideo);
    remaining.splice(remaining.indexOf(nextVideo), 1);
  }

  return diversified;
}

/**
 * コンテンツメトリクスの計算
 * Calculate content metrics
 */
export function calculateContentMetrics(videos: VideoWithRelations[]): DiversityMetrics {
  if (videos.length === 0) {
    return {
      genre_diversity: 0,
      maker_diversity: 0,
      price_range: { min: 0, max: 0, avg: 0 },
      content_freshness: 0
    };
  }

  const genres = new Set(videos.map(v => v.genre));
  const makers = new Set(videos.map(v => v.maker));
  const prices = videos.map(v => v.price).filter(p => p > 0);

  const priceMin = prices.length > 0 ? Math.min(...prices) : 0;
  const priceMax = prices.length > 0 ? Math.max(...prices) : 0;
  const priceAvg = prices.length > 0 ? prices.reduce((a, b) => a + b, 0) / prices.length : 0;

  // Calculate freshness based on creation dates
  const now = new Date();
  const freshness = videos.reduce((total, video) => {
    if (!video.created_at) return total;
    const created = new Date(video.created_at);
    const daysSinceCreated = (now.getTime() - created.getTime()) / (1000 * 60 * 60 * 24);
    return total + Math.max(0, 1 - (daysSinceCreated / 365)); // Freshness decreases over a year
  }, 0) / videos.length;

  return {
    genre_diversity: genres.size / Math.max(videos.length, 1),
    maker_diversity: makers.size / Math.max(videos.length, 1),
    price_range: {
      min: priceMin,
      max: priceMax,
      avg: priceAvg
    },
    content_freshness: freshness
  };
}

/**
 * コンテンツの推薦理由を生成
 * Generate recommendation reason for content
 */
export function generateRecommendationReason(
  video: VideoWithRelations,
  context: {
    similarity_score?: number;
    algorithm?: string;
    user_preferences?: string[];
  } = {}
): string {
  const reasons = [];

  if (context.similarity_score) {
    if (context.similarity_score > 0.8) {
      reasons.push('Perfect match for your taste');
    } else if (context.similarity_score > 0.6) {
      reasons.push('Similar to your liked videos');
    } else if (context.similarity_score > 0.4) {
      reasons.push('You might like this');
    }
  }

  if (context.algorithm === 'trending') {
    reasons.push('Currently trending');
  } else if (context.algorithm === 'new') {
    reasons.push('Recently added');
  } else if (context.algorithm === 'popular') {
    reasons.push('Popular with users');
  }

  if (video.performers && video.performers.length > 0 && context.user_preferences) {
    const matchingPerformers = video.performers.filter(p =>
      context.user_preferences!.some(pref =>
        p.toLowerCase().includes(pref.toLowerCase())
      )
    );
    if (matchingPerformers.length > 0) {
      reasons.push(`Features ${matchingPerformers[0]}`);
    }
  }

  if (video.genre && context.user_preferences?.includes(video.genre)) {
    reasons.push(`${video.genre} content you enjoy`);
  }

  return reasons.length > 0 ? reasons[0] : 'Recommended for you';
}

/**
 * パフォーマンス測定ユーティリティ
 * Performance measurement utility
 */
export function measurePerformance<T>(
  operationName: string,
  operation: () => Promise<T>
): Promise<{ result: T; metrics: PerformanceMetrics }> {
  const startTime = performance.now();
  const startMemory = performance.memory?.usedJSHeapSize || 0;

  return operation().then(result => {
    const endTime = performance.now();
    const endMemory = performance.memory?.usedJSHeapSize || 0;

    return {
      result,
      metrics: {
        operation_name: operationName,
        duration_ms: endTime - startTime,
        memory_usage: endMemory - startMemory,
        timestamp: new Date().toISOString()
      }
    };
  });
}

/**
 * コンテンツキャッシュ管理
 * Content cache management
 */
export class ContentCache {
  private static cache = new Map<string, { data: any; expiry: number }>();

  static set(key: string, data: any, ttlMs: number = 300000): void {
    this.cache.set(key, {
      data,
      expiry: Date.now() + ttlMs
    });
  }

  static get<T>(key: string): T | null {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }

    return item.data as T;
  }

  static clear(): void {
    this.cache.clear();
  }

  static cleanup(): void {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiry) {
        this.cache.delete(key);
      }
    }
  }
}

/**
 * バッチ処理のためのコンテンツ取得
 * Batch content fetching for processing
 */
export async function batchGetVideos(
  supabase: SupabaseClient,
  batchSize: number = 100,
  offset: number = 0,
  filters?: ContentFilters
): Promise<DatabaseResult<VideoWithRelations[]>> {
  try {
    let query = supabase
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        sample_video_url,
        maker,
        price,
        duration_seconds,
        image_urls,
        created_at,
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .range(offset, offset + batchSize - 1)
      .order('created_at', { ascending: false });

    if (filters) {
      if (filters.genres && filters.genres.length > 0) {
        query = query.in('id',
          supabase
            .from('video_tags')
            .select('video_id')
            .in('tag_id',
              supabase
                .from('tags')
                .select('id')
                .in('name', filters.genres)
            )
        );
      }
      if (filters.content_ids && filters.content_ids.length > 0) {
        query = query.not('id', 'in', filters.content_ids);
      }
    }

    const { data: videos, error } = await query;

    if (error) {
      return handleDatabaseError(error);
    }

    const formattedVideos: VideoWithRelations[] = videos?.map((video: any) => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

      return {
        id: video.id,
        title: video.title,
        description: video.description || '',
        thumbnail_url: video.thumbnail_url,
        preview_video_url: video.preview_video_url,
        sample_video_url: video.sample_video_url,
        maker: video.maker,
        genre: genreTag, // tagsから動的に生成されたgenre
        price: video.price || 0,
        duration_seconds: video.duration_seconds,
        image_urls: video.image_urls || [],
        performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        tags: tags,
        created_at: video.created_at
      };
    }) || [];

    return createSuccessResult(formattedVideos);

  } catch (error) {
    return handleDatabaseError(error);
  }
}

// Export for backward compatibility
export type { VideoWithRelations, ContentFilters, DiversityMetrics, PerformanceMetrics };
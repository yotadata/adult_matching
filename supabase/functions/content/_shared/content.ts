/**
 * コンテンツ管理系Edge Functions共通ユーティリティ
 */

import { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

export interface VideoContent {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
  created_at?: string;
  similarity_score?: number;
  recommendation_reason?: string;
}

export interface ContentFilters {
  genres?: string[];
  makers?: string[];
  performers?: string[];
  tags?: string[];
  price_min?: number;
  price_max?: number;
  content_ids?: string[];
}

export interface PaginationParams {
  limit: number;
  offset: number;
  cursor?: string;
}

/**
 * 動画コンテンツを取得（パフォーマンス最適化版）
 */
export async function getVideoContent(
  supabase: SupabaseClient,
  videoIds: string[],
  includeEmbeddings: boolean = false
): Promise<VideoContent[]> {
  try {
    const selectFields = `
      id,
      title,
      description,
      thumbnail_url,
      preview_video_url,
      maker,
      genre,
      price,
      sample_video_url,
      image_urls,
      created_at,
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
      ${includeEmbeddings ? ', embedding' : ''}
    `;

    const { data: videos, error } = await supabase
      .from('videos')
      .select(selectFields)
      .in('id', videoIds);

    if (error) {
      console.error('Error fetching video content:', error);
      throw new Error('Failed to fetch video content');
    }

    return videos?.map(formatVideoContent) || [];
  } catch (error) {
    console.error('Error in getVideoContent:', error);
    throw error;
  }
}

/**
 * 動画コンテンツをフォーマット
 */
function formatVideoContent(video: any): VideoContent {
  return {
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: video.genre,
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
    tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    created_at: video.created_at,
    similarity_score: video.similarity_score,
    recommendation_reason: video.recommendation_reason
  };
}

/**
 * フィルタ条件に基づいて動画を検索
 */
export async function searchVideosWithFilters(
  supabase: SupabaseClient,
  filters: ContentFilters,
  pagination: PaginationParams
): Promise<{ videos: VideoContent[]; total_count: number }> {
  try {
    let query = supabase
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        maker,
        genre,
        price,
        sample_video_url,
        image_urls,
        created_at,
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `, { count: 'exact' });

    // フィルタ適用
    if (filters.genres?.length) {
      query = query.in('genre', filters.genres);
    }
    
    if (filters.makers?.length) {
      query = query.in('maker', filters.makers);
    }
    
    if (filters.price_min !== undefined) {
      query = query.gte('price', filters.price_min);
    }
    
    if (filters.price_max !== undefined) {
      query = query.lte('price', filters.price_max);
    }

    if (filters.content_ids?.length) {
      query = query.in('id', filters.content_ids);
    }

    // ソートとページネーション
    query = query
      .order('created_at', { ascending: false })
      .range(pagination.offset, pagination.offset + pagination.limit - 1);

    const { data: videos, error, count } = await query;

    if (error) {
      console.error('Error searching videos:', error);
      throw new Error('Failed to search videos');
    }

    return {
      videos: videos?.map(formatVideoContent) || [],
      total_count: count || 0
    };
  } catch (error) {
    console.error('Error in searchVideosWithFilters:', error);
    throw error;
  }
}

/**
 * ランダムな動画を取得（探索フィード用）
 */
export async function getRandomVideos(
  supabase: SupabaseClient,
  limit: number,
  excludeIds: string[] = []
): Promise<VideoContent[]> {
  try {
    let query = supabase
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        maker,
        genre,
        price,
        sample_video_url,
        image_urls,
        created_at,
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .limit(limit * 3); // 多めに取得してランダム性を確保

    if (excludeIds.length > 0) {
      query = query.not('id', 'in', `(${excludeIds.join(',')})`);
    }

    const { data: videos, error } = await query;

    if (error) {
      console.error('Error getting random videos:', error);
      throw new Error('Failed to get random videos');
    }

    if (!videos || videos.length === 0) {
      return [];
    }

    // ランダムにシャッフルして指定数だけ返す
    const shuffled = videos.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, limit).map(formatVideoContent);
  } catch (error) {
    console.error('Error in getRandomVideos:', error);
    throw error;
  }
}

/**
 * ベクトル類似度検索（推薦システム用）
 */
export async function getVideosBySimilarity(
  supabase: SupabaseClient,
  userEmbedding: number[],
  limit: number,
  excludeIds: string[] = []
): Promise<VideoContent[]> {
  try {
    // PostgreSQLのベクトル類似度検索を使用
    let rpcQuery = supabase.rpc('match_videos_cosine', {
      query_embedding: userEmbedding,
      match_threshold: 0.1,
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

    // 除外IDsをフィルタ
    const filteredMatches = matches.filter((match: any) => 
      !excludeIds.includes(match.id)
    );

    // 動画詳細情報を取得
    const videoIds = filteredMatches.map((match: any) => match.id);
    const videos = await getVideoContent(supabase, videoIds);

    // 類似度スコアを追加
    return videos.map(video => {
      const match = filteredMatches.find((m: any) => m.id === video.id);
      return {
        ...video,
        similarity_score: match?.similarity || 0
      };
    });
  } catch (error) {
    console.error('Error in getVideosBySimilarity:', error);
    throw error;
  }
}

/**
 * コンテンツの多様性を確保するフィルタ
 */
export function diversifyContent(
  videos: VideoContent[],
  maxPerGenre: number = 3,
  maxPerMaker: number = 2
): VideoContent[] {
  const genreCounts: Record<string, number> = {};
  const makerCounts: Record<string, number> = {};
  
  return videos.filter(video => {
    const genreCount = genreCounts[video.genre] || 0;
    const makerCount = makerCounts[video.maker] || 0;
    
    if (genreCount >= maxPerGenre || makerCount >= maxPerMaker) {
      return false;
    }
    
    genreCounts[video.genre] = genreCount + 1;
    makerCounts[video.maker] = makerCount + 1;
    
    return true;
  });
}

/**
 * コンテンツのパフォーマンス指標を計算
 */
export function calculateContentMetrics(videos: VideoContent[]): {
  genre_diversity: number;
  maker_diversity: number;
  price_range: { min: number; max: number; avg: number };
  content_freshness: number;
} {
  if (videos.length === 0) {
    return {
      genre_diversity: 0,
      maker_diversity: 0,
      price_range: { min: 0, max: 0, avg: 0 },
      content_freshness: 0
    };
  }

  const uniqueGenres = new Set(videos.map(v => v.genre)).size;
  const uniqueMakers = new Set(videos.map(v => v.maker)).size;
  
  const prices = videos.map(v => v.price);
  const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
  
  // 新しさ指標（過去30日以内のコンテンツの割合）
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  
  const recentContent = videos.filter(v => 
    v.created_at && new Date(v.created_at) > thirtyDaysAgo
  ).length;
  
  const freshness = recentContent / videos.length;

  return {
    genre_diversity: uniqueGenres / videos.length,
    maker_diversity: uniqueMakers / videos.length,
    price_range: {
      min: Math.min(...prices),
      max: Math.max(...prices),
      avg: avgPrice
    },
    content_freshness: freshness
  };
}
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import {
  corsHeaders,
  optionsResponse,
  errorResponse,
  successResponse
} from "../../_shared/responses.ts";
import {
  getSupabaseClientFromRequest,
  DatabaseResult
} from "../../_shared/database.ts";

// Inline type definition to avoid import issues
interface VideoWithTags {
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
  genre: string;
  all_tags: string[];
  created_at?: string;
  updated_at?: string;
}

interface VideosFeedRequest {
  page?: number;
  limit?: number;
  personalized?: boolean;
  phase?: 'discovery' | 'recommendation';
  exclude_seen?: boolean;
  genre_filter?: string[];
  maker_filter?: string[];
}

interface VideosFeedResponse {
  videos: VideoItem[];
  pagination: {
    page: number;
    limit: number;
    total_count?: number;
    has_more: boolean;
  };
  feed_type: 'personalized' | 'discovery' | 'popular';
  recommendation_meta?: {
    user_embedding_found: boolean;
    similarity_threshold: number;
    phase: string;
  };
}

interface VideoItem extends VideoWithTags {
  rating?: number;
  performers: string[];
  tags: string[];
  recommendation_reason?: string;
}

export default serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  if (req.method !== 'POST') {
    return errorResponse('Method not allowed', 405, 'METHOD_NOT_ALLOWED');
  }

  try {
    const supabaseClient = getSupabaseClientFromRequest(req);
    
    // リクエストパラメータの解析
    const params: VideosFeedRequest = await req.json().catch(() => ({}));
    const {
      page = 1,
      limit = 20,
      personalized = true,
      phase = 'discovery',
      exclude_seen = true,
      genre_filter = [],
      maker_filter = []
    } = params;

    // ユーザー認証の確認（オプショナル）
    let userId: string | null = null;
    try {
      const authHeader = req.headers.get('Authorization');
      if (authHeader) {
        const { data: { user }, error } = await supabaseClient.auth.getUser(
          authHeader.replace('Bearer ', '')
        );
        if (!error && user) {
          userId = user.id;
        }
      }
    } catch (error) {
      console.log('No valid authentication provided, using discovery feed');
    }

    console.log(`Fetching videos feed - User: ${userId || 'anonymous'}, Phase: ${phase}, Personalized: ${personalized}`);

    let feedResult: DatabaseResult<VideosFeedResponse>;

    if (userId && personalized) {
      // 個人化フィードの取得
      feedResult = await getPersonalizedFeed(supabaseClient, userId, {
        page,
        limit,
        phase,
        exclude_seen,
        genre_filter,
        maker_filter
      });
    } else {
      // 発見フィードの取得
      feedResult = await getDiscoveryFeed(supabaseClient, {
        page,
        limit,
        genre_filter,
        maker_filter
      });
    }

    if (!feedResult.success) {
      return errorResponse(feedResult.error || 'Failed to fetch videos feed', 500, 'FEED_ERROR');
    }

    return successResponse(feedResult.data, 'Videos feed retrieved successfully');

  } catch (error) {
    console.error('Videos feed error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
});

// 個人化フィードの取得
async function getPersonalizedFeed(
  supabaseClient: any,
  userId: string,
  options: {
    page: number;
    limit: number;
    phase: string;
    exclude_seen: boolean;
    genre_filter: string[];
    maker_filter: string[];
  }
): Promise<DatabaseResult<VideosFeedResponse>> {
  try {
    const { page, limit, phase, exclude_seen, genre_filter, maker_filter } = options;
    const offset = (page - 1) * limit;

    // ユーザーエンベディングの存在確認
    const { data: userEmbedding, error: embeddingError } = await supabaseClient
      .from('user_embeddings')
      .select('embedding, updated_at')
      .eq('user_id', userId)
      .single();

    if (embeddingError || !userEmbedding) {
      console.log(`No user embedding found for user ${userId}, falling back to discovery feed`);
      return await getDiscoveryFeed(supabaseClient, { page, limit, genre_filter, maker_filter });
    }

    // 類似度ベースの推薦を実行（tags-basedフィルタを含む）
    const { data: recommendations, error: recError } = await supabaseClient
      .rpc('get_personalized_recommendations', {
        p_user_id: userId,
        p_limit: limit * 2, // より多くの候補を取得してフィルタリング
        p_offset: offset,
        p_similarity_threshold: phase === 'discovery' ? 0.3 : 0.5,
        p_exclude_seen: exclude_seen,
        p_genre_filter: genre_filter.length > 0 ? genre_filter : null,
        p_maker_filter: maker_filter.length > 0 ? maker_filter : null
      });

    if (recError) {
      console.error('Personalization RPC error:', recError);
      // フォールバック: 発見フィードを使用
      return await getDiscoveryFeed(supabaseClient, { page, limit, genre_filter, maker_filter });
    }

    if (!recommendations || recommendations.length === 0) {
      // 推薦結果がない場合はハイブリッドフィードを試行
      return await getHybridFeed(supabaseClient, userId, options);
    }

    // 結果の整形（tags-basedデータ使用）
    const videos: VideoItem[] = recommendations.slice(0, limit).map((video: any) => ({
      id: video.id,
      title: video.title,
      description: video.description || '',
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      maker: video.maker,
      genre: video.genre || 'Unknown', // RPCまたはtagsから取得されたgenre
      price: video.price || 0,
      sample_video_url: video.sample_video_url,
      image_urls: video.image_urls || [],
      duration_seconds: video.duration_seconds,
      rating: video.rating,
      performers: video.performers || [],
      tags: video.tags || [],
      all_tags: video.all_tags || [],
      similarity_score: video.similarity_score,
      recommendation_reason: generateRecommendationReason(video, phase),
      created_at: video.created_at,
      updated_at: video.updated_at
    }));

    const response: VideosFeedResponse = {
      videos,
      pagination: {
        page,
        limit,
        has_more: recommendations.length > limit
      },
      feed_type: 'personalized',
      recommendation_meta: {
        user_embedding_found: true,
        similarity_threshold: phase === 'discovery' ? 0.3 : 0.5,
        phase
      }
    };

    return { success: true, data: response };

  } catch (error) {
    console.error('Personalized feed error:', error);
    return { success: false, error: 'Failed to generate personalized feed' };
  }
}

// 発見フィードの取得
async function getDiscoveryFeed(
  supabaseClient: any,
  options: {
    page: number;
    limit: number;
    genre_filter: string[];
    maker_filter: string[];
  }
): Promise<DatabaseResult<VideosFeedResponse>> {
  try {
    const { page, limit, genre_filter, maker_filter } = options;
    const offset = (page - 1) * limit;

    // 人気とランダムのハイブリッド発見フィード
    let query = supabaseClient
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        maker,
        price,
        sample_video_url,
        image_urls,
        duration_seconds,
        created_at,
        updated_at,
        rating,
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `);

    // フィルタの適用
    if (genre_filter.length > 0) {
      query = query.in('id',
        supabaseClient
          .from('video_tags')
          .select('video_id')
          .in('tag_id',
            supabaseClient
              .from('tags')
              .select('id')
              .in('name', genre_filter)
          )
      );
    }
    
    if (maker_filter.length > 0) {
      query = query.in('maker', maker_filter);
    }

    // ランダムソートと制限
    const { data: videos, error } = await query
      .order('rating', { ascending: false, nullsLast: true })
      .range(offset, offset + limit - 1);

    if (error) {
      throw new Error(`Discovery feed query error: ${error.message}`);
    }

    // 結果の整形（tags-basedデータ使用）
    const formattedVideos: VideoItem[] = (videos || []).map((video: any) => ({
      id: video.id,
      title: video.title,
      description: video.description || '',
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      maker: video.maker,
      genre: (() => {
        const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
        return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
      })(), // tagsから動的に生成されたgenre
      price: video.price || 0,
      sample_video_url: video.sample_video_url,
      image_urls: video.image_urls || [],
      duration_seconds: video.duration_seconds,
      rating: video.rating,
      performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
      all_tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [], // 全タグをall_tagsとして使用
      recommendation_reason: 'Popular content',
      created_at: video.created_at,
      updated_at: video.updated_at
    }));

    const response: VideosFeedResponse = {
      videos: formattedVideos,
      pagination: {
        page,
        limit,
        has_more: formattedVideos.length === limit
      },
      feed_type: 'discovery'
    };

    return { success: true, data: response };

  } catch (error) {
    console.error('Discovery feed error:', error);
    return { success: false, error: 'Failed to generate discovery feed' };
  }
}

// ハイブリッドフィードの取得（個人化 + 発見）
async function getHybridFeed(
  supabaseClient: any,
  userId: string,
  options: {
    page: number;
    limit: number;
    phase: string;
    exclude_seen: boolean;
    genre_filter: string[];
    maker_filter: string[];
  }
): Promise<DatabaseResult<VideosFeedResponse>> {
  try {
    const { limit } = options;
    
    // 個人化40% + 発見60%の比率でハイブリッド
    const personalizedCount = Math.floor(limit * 0.4);
    const discoveryCount = limit - personalizedCount;

    // 並行して取得
    const [personalizedResult, discoveryResult] = await Promise.all([
      getPersonalizedFeed(supabaseClient, userId, {
        ...options,
        limit: personalizedCount
      }),
      getDiscoveryFeed(supabaseClient, {
        ...options,
        limit: discoveryCount
      })
    ]);

    const combinedVideos: VideoItem[] = [];
    
    if (personalizedResult.success && personalizedResult.data) {
      combinedVideos.push(...personalizedResult.data.videos);
    }
    
    if (discoveryResult.success && discoveryResult.data) {
      combinedVideos.push(...discoveryResult.data.videos);
    }

    // 重複を除去し、シャッフル
    const uniqueVideos = Array.from(
      new Map(combinedVideos.map(video => [video.id, video])).values()
    ).slice(0, limit);

    const response: VideosFeedResponse = {
      videos: uniqueVideos,
      pagination: options,
      feed_type: 'personalized',
      recommendation_meta: {
        user_embedding_found: true,
        similarity_threshold: 0.4,
        phase: 'hybrid'
      }
    };

    return { success: true, data: response };

  } catch (error) {
    console.error('Hybrid feed error:', error);
    return { success: false, error: 'Failed to generate hybrid feed' };
  }
}

// 推薦理由の生成
function generateRecommendationReason(video: any, phase: string): string {
  const reasons = [];

  if (video.similarity_score > 0.7) {
    reasons.push('Highly matches your preferences');
  } else if (video.similarity_score > 0.5) {
    reasons.push('Similar to your liked videos');
  }

  if (phase === 'discovery') {
    reasons.push('Discover new content');
  }

  if (video.rating > 4) {
    reasons.push('Highly rated');
  }

  return reasons.length > 0 ? reasons[0] : 'Recommended for you';
}
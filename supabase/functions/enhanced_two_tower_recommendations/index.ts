import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

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

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface EnhancedRecommendedVideo extends VideoWithTags {
  preview_video_url: string;
  sample_video_url: string;
  performers: string[];
  tags: string[];
  similarity_score: number;
  recommendation_reason: string;
  diversity_score?: number;
  confidence_score?: number;
}

interface UserEmbedding {
  user_id: string;
  embedding: number[];
  updated_at: string;
}

interface VideoEmbedding {
  video_id: string;
  embedding: number[];
  updated_at: string;
}

interface RecommendationRequest {
  limit?: number;
  exclude_liked?: boolean;
  diversity_weight?: number;
  include_reasons?: boolean;
  min_similarity_threshold?: number;
}

interface RecommendationMetrics {
  total_candidates: number;
  embedding_hits: number;
  similarity_computation_time: number;
  total_processing_time: number;
  diversity_score: number;
  avg_confidence: number;
}

/**
 * 768次元ベクター間のコサイン類似度計算（最適化版）
 */
function calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length || vec1.length !== 768) {
    throw new Error('Both vectors must be 768-dimensional');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  // SIMD風最適化：4つずつ処理
  for (let i = 0; i < 768; i += 4) {
    const a1 = vec1[i], a2 = vec1[i+1] || 0, a3 = vec1[i+2] || 0, a4 = vec1[i+3] || 0;
    const b1 = vec2[i], b2 = vec2[i+1] || 0, b3 = vec2[i+2] || 0, b4 = vec2[i+3] || 0;

    dotProduct += a1*b1 + a2*b2 + a3*b3 + a4*b4;
    norm1 += a1*a1 + a2*a2 + a3*a3 + a4*a4;
    norm2 += b1*b1 + b2*b2 + b3*b3 + b4*b4;
  }

  const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);
  return magnitude > 0 ? dotProduct / magnitude : 0;
}

/**
 * 高速バッチ類似度計算
 */
function batchCalculateSimilarities(
  userEmbedding: number[], 
  videoEmbeddings: VideoEmbedding[]
): { video_id: string, similarity: number }[] {
  return videoEmbeddings.map(video => ({
    video_id: video.video_id,
    similarity: calculateCosineSimilarity(userEmbedding, video.embedding)
  }));
}

/**
 * 推薦理由の生成
 */
function generateRecommendationReason(
  video: any,
  similarity: number,
  userGenrePreferences: Record<string, number>
): string {
  const reasons: string[] = [];
  
  // 類似度ベースの理由
  if (similarity > 0.9) {
    reasons.push("高精度マッチング");
  } else if (similarity > 0.8) {
    reasons.push("良好なマッチング");
  } else if (similarity > 0.7) {
    reasons.push("適度なマッチング");
  }

  // ジャンル嗜好に基づく理由
  const userGenreScore = userGenrePreferences[video.genre] || 0;
  if (userGenreScore > 0.3) {
    reasons.push(`好みのジャンル（${video.genre}）`);
  }

  // メーカー、価格帯による理由
  if (video.price && video.price < 3000) {
    reasons.push("お手頃価格");
  } else if (video.price && video.price > 5000) {
    reasons.push("プレミアムコンテンツ");
  }

  // パフォーマー要素
  if (video.performers && video.performers.length > 0) {
    const topPerformer = video.performers[0];
    if (topPerformer) {
      reasons.push(`人気パフォーマー出演`);
    }
  }

  return reasons.length > 0 
    ? reasons.join(" | ") + ` (類似度: ${(similarity * 100).toFixed(1)}%)`
    : `AI推薦 (類似度: ${(similarity * 100).toFixed(1)}%)`;
}

/**
 * 推薦結果の多様性向上
 */
function ensureGenreDiversity(
  recommendations: any[],
  limit: number
): any[] {
  if (recommendations.length <= 1) return recommendations;

  // ジャンル、メーカー、価格帯による多様性スコア計算
  const genreCount = new Map<string, number>();
  const makerCount = new Map<string, number>();
  const priceRanges = new Map<string, number>();

  recommendations.forEach(rec => {
    genreCount.set(rec.genre, (genreCount.get(rec.genre) || 0) + 1);
    makerCount.set(rec.maker, (makerCount.get(rec.maker) || 0) + 1);
    
    const priceRange = rec.price < 3000 ? 'low' : rec.price < 5000 ? 'mid' : 'high';
    priceRanges.set(priceRange, (priceRanges.get(priceRange) || 0) + 1);
  });

  // 多様性スコアの計算と適用
  const diversityEnhanced = recommendations.map((rec, index) => {
    const genrePenalty = (genreCount.get(rec.genre) || 1) / recommendations.length;
    const makerPenalty = (makerCount.get(rec.maker) || 1) / recommendations.length;
    const priceRange = rec.price < 3000 ? 'low' : rec.price < 5000 ? 'mid' : 'high';
    const pricePenalty = (priceRanges.get(priceRange) || 1) / recommendations.length;

    const diversityPenalty = (genrePenalty + makerPenalty + pricePenalty) / 3;
    const diversityScore = 1 - diversityPenalty;
    
    // 類似度スコアと多様性スコアの重み付き組み合わせ
    const combinedScore = 0.7 * rec.similarity_score + 0.3 * diversityScore;

    return {
      ...rec,
      diversity_score: diversityScore,
      final_score: combinedScore
    };
  }).sort((a, b) => b.final_score - a.final_score);

  return diversityEnhanced.slice(0, limit);
}

/**
 * ユーザーの埋め込み取得（フォールバック付き）
 */
async function getUserEmbeddingWithFallback(
  supabaseClient: any,
  userId: string
): Promise<number[] | null> {
  try {
    // データベースから既存の埋め込みを取得
    const { data: userEmbedding, error } = await supabaseClient
      .from('user_embeddings')
      .select('embedding')
      .eq('user_id', userId)
      .single();

    if (userEmbedding && !error) {
      console.log(`User embedding found for ${userId}`);
      return userEmbedding.embedding;
    }

    console.log(`No user embedding found for ${userId}, generating default`);
    
    // フォールバック：ユーザー行動から特徴量を生成
    const { data: userBehavior } = await supabaseClient
      .from('likes')
      .select(`
        created_at,
        videos!inner(
          id,
          maker,
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(50);

    if (userBehavior && userBehavior.length > 0) {
      // ユーザー行動から簡易埋め込み生成
      return generateUserEmbeddingFromBehavior(userBehavior.map(b => {
        const tags = b.videos.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
        const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
        return {
          decision_type: 'like',
          videos: { genre: genreTag }
        };
      }));
    }

    // 最終フォールバック：デフォルト埋め込み
    return generateDefaultUserEmbedding();

  } catch (error) {
    console.error('Error fetching user embedding:', error);
    return generateDefaultUserEmbedding();
  }
}

/**
 * ユーザー行動からの簡易埋め込み生成
 */
function generateUserEmbeddingFromBehavior(behavior: any[]): number[] {
  const embedding = new Array(768).fill(0);
  
  // ジャンル嗜好（最初の100次元）
  const genrePrefs = new Map<string, number>();
  const likeCount = behavior.filter(b => b.decision_type === 'like').length;
  
  behavior.forEach(b => {
    if (b.decision_type === 'like' && b.videos.genre) {
      genrePrefs.set(b.videos.genre, (genrePrefs.get(b.videos.genre) || 0) + 1);
    }
  });

  let dimIndex = 0;
  genrePrefs.forEach((count, genre) => {
    if (dimIndex < 100) {
      embedding[dimIndex] = count / Math.max(1, likeCount);
      dimIndex++;
    }
  });

  // 価格嗜好（次の50次元）
  const prices = behavior
    .filter(b => b.decision_type === 'like' && b.videos.price)
    .map(b => b.videos.price);
  
  if (prices.length > 0) {
    const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const priceStd = Math.sqrt(prices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / prices.length);
    
    for (let i = 100; i < 150; i++) {
      embedding[i] = (avgPrice / 10000) * Math.sin((i - 100) * Math.PI / 50);
    }
  }

  // メーカー嗜好（次の100次元）
  const makerPrefs = new Map<string, number>();
  behavior.forEach(b => {
    if (b.decision_type === 'like' && b.videos.maker) {
      makerPrefs.set(b.videos.maker, (makerPrefs.get(b.videos.maker) || 0) + 1);
    }
  });

  dimIndex = 150;
  makerPrefs.forEach((count, maker) => {
    if (dimIndex < 250) {
      embedding[dimIndex] = count / Math.max(1, likeCount);
      dimIndex++;
    }
  });

  // 残りの次元：ランダムノイズで多様性確保
  for (let i = 250; i < 768; i++) {
    embedding[i] = (Math.random() - 0.5) * 0.1;
  }

  // L2正規化
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / (norm || 1));
}

/**
 * デフォルトユーザー埋め込み生成
 */
function generateDefaultUserEmbedding(): number[] {
  const embedding = new Array(768).fill(0);
  
  // 平均的な嗜好を表現
  for (let i = 0; i < 768; i++) {
    embedding[i] = (Math.random() - 0.5) * 0.2;
  }

  // L2正規化
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / (norm || 1));
}

/**
 * ユーザーのジャンル嗜好分析
 */
async function analyzeUserGenrePreferences(
  supabaseClient: any,
  userId: string
): Promise<Record<string, number>> {
  try {
    // ユーザーのジャンル嗜好を分析
    const { data: likes, error } = await supabaseClient
      .from('likes')
      .select(`
        created_at,
        videos!inner(
          id,
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error || !likes) {
      console.warn('Failed to get user genre preferences:', error);
      return {};
    }

    const genreCount = new Map<string, number>();
    const totalLikes = likes.length;

    likes.forEach(like => {
      const tags = like.videos.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
      if (genreTag && genreTag !== 'Unknown') {
        genreCount.set(genreTag, (genreCount.get(genreTag) || 0) + 1);
      }
    });

    const preferences: Record<string, number> = {};
    genreCount.forEach((count, genre) => {
      preferences[genre] = count / Math.max(1, totalLikes);
    });

    return preferences;
  } catch (error) {
    console.error('Error analyzing user genre preferences:', error);
    return {};
  }
}

serve(async (req: Request) => {
  // CORS対応
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const startTime = performance.now();
  const metrics: RecommendationMetrics = {
    total_candidates: 0,
    embedding_hits: 0,
    similarity_computation_time: 0,
    total_processing_time: 0,
    diversity_score: 0,
    avg_confidence: 0
  };

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const { user_id, limit = 20, min_similarity_threshold = 0.1, exclude_liked = true } = await req.json();

    if (!user_id) {
      return new Response(
        JSON.stringify({ error: 'user_id is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // ユーザー埋め込みベクトルの取得または生成
    const userEmbedding = await getUserEmbeddingWithFallback(supabaseClient, user_id);
    if (!userEmbedding) {
      throw new Error('Failed to get user embedding');
    }


    // 候補動画の取得（tags-basedクエリ使用）
    let videoQuery = supabaseClient
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        maker,
        price,
        duration_seconds,
        sample_video_url,
        image_urls,
        video_performers!inner(performers(name)),
        video_tags!inner(tags(name))
      `)
      .limit(1000);

    // いいね済み動画の除外
    if (exclude_liked) {
      const { data: likedVideos } = await supabaseClient
        .from('likes')
        .select('video_id')
        .eq('user_id', user_id);
      
      const likedIds = likedVideos?.map(l => l.video_id) || [];
      if (likedIds.length > 0) {
        videoQuery = videoQuery.not('id', 'in', `(${likedIds.join(',')})`)
      }
    }

    const { data: candidateVideos, error: videosError } = await videoQuery;
    if (videosError) {
      throw new Error(`Failed to fetch videos: ${videosError.message}`);
    }

    metrics.total_candidates = candidateVideos?.length || 0;

    if (!candidateVideos?.length) {
      return new Response(
        JSON.stringify({ 
          recommendations: [],
          message: 'No candidate videos found',
          metrics
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // ビデオ埋め込みの取得
    const candidateVideoIds = candidateVideos.map(v => v.id);
    const { data: videoEmbeddings, error: embeddingError } = await supabaseClient
      .from('video_embeddings')
      .select('video_id, embedding')
      .in('video_id', candidateVideoIds);

    if (embeddingError) {
      console.error('Error fetching video embeddings:', embeddingError);
    }

    const embeddingMap = new Map<string, number[]>();
    videoEmbeddings?.forEach(ve => {
      embeddingMap.set(ve.video_id, ve.embedding);
    });

    metrics.embedding_hits = embeddingMap.size;

    // 埋め込みがあるビデオのみでフィルタリング
    const videosWithEmbeddings = candidateVideos.filter(video => 
      embeddingMap.has(video.id)
    );

    if (videosWithEmbeddings.length === 0) {
      return new Response(
        JSON.stringify({ 
          recommendations: [],
          message: 'No videos with embeddings found',
          metrics
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // 類似度計算
    const similarityStartTime = performance.now();
    
    const videoEmbeddingList = videosWithEmbeddings.map(video => ({
      video_id: video.id,
      embedding: embeddingMap.get(video.id)!
    }));

    const similarities = batchCalculateSimilarities(userEmbedding, videoEmbeddingList);
    
    metrics.similarity_computation_time = performance.now() - similarityStartTime;

    // 類似度によるフィルタリングとソート
    const filteredSimilarities = similarities
      .filter(sim => sim.similarity >= min_similarity_threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit * 2); // 多様性確保のため多めに取得

    // 推薦結果の構築（tags-basedデータ使用）
    let recommendations: EnhancedRecommendedVideo[] = filteredSimilarities.map(sim => {
      const video = videosWithEmbeddings.find(v => v.id === sim.video_id)!;
      
      return {
        id: video.id,
        title: video.title,
        description: video.description,
        thumbnail_url: video.thumbnail_url,
        preview_video_url: video.preview_video_url,
        maker: video.maker,
        genre: (() => {
          const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
          return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
        })(), // tagsから動的に生成されたgenre
        price: video.price,
        sample_video_url: video.sample_video_url,
        image_urls: video.image_urls || [],
        performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        similarity_score: Math.round(sim.similarity * 100) / 100,
        tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [], // 全タグをall_tagsとして使用
        all_tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
        recommendation_reason: generateRecommendationReason(video, sim.similarity, {}),
        confidence_score: sim.similarity,
        created_at: video.created_at,
        updated_at: video.updated_at
      };
    });

    // 多様性フィルタリング
    recommendations = ensureGenreDiversity(recommendations, limit);

    // パフォーマンスメトリクスの完了
    metrics.total_processing_time = performance.now() - startTime;
    metrics.diversity_score = recommendations.length > 1
      ? new Set(recommendations.map(r => r.genre)).size / recommendations.length
      : 0;
    metrics.avg_confidence = recommendations.length > 0
      ? recommendations.reduce((sum, r) => sum + (r.confidence_score || 0), 0) / recommendations.length
      : 0;

    // 統計情報の生成
    const genreDistribution = recommendations.reduce((acc, rec) => {
      acc[rec.genre] = (acc[rec.genre] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const response = {
      recommendations,
      total_candidates: metrics.total_candidates,
      metrics,
      diversity_metrics: {
        genre_distribution: genreDistribution,
        unique_genres: Object.keys(genreDistribution).length,
        genre_preferences: Object.entries(await analyzeUserGenrePreferences(supabaseClient, user_id))
          .sort(([,a], [,b]) => b - a)
          .slice(0, 5)
          .map(([genre, score]) => ({ genre, score: Math.round(score * 100) / 100 }))
      }
    };

    return new Response(JSON.stringify(response), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in enhanced_two_tower_recommendations:', error);
    
    metrics.total_processing_time = performance.now() - startTime;

    return new Response(
      JSON.stringify({
        error: error.message,
        metrics
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
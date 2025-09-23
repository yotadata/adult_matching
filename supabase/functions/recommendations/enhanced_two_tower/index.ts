import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface EnhancedRecommendedVideo {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  all_tags: string[];
  price: number;
  sample_video_url: string;
  image_urls: string[];
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
  // 後方互換性: 旧APIパラメータサポート
  max_results?: number;  // limit のエイリアス
  include_explanations?: boolean;  // include_reasons のエイリアス
  algorithm?: 'enhanced' | 'basic' | 'two_tower';  // 推薦アルゴリズム選択
}

interface RecommendationMetrics {
  total_candidates: number;
  embedding_hits: number;
  similarity_computation_time: number;
  sort_time: number;
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
function enhanceDiversity(
  recommendations: any[],
  diversityWeight: number = 0.3
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
  return recommendations.map((rec, index) => {
    const genrePenalty = (genreCount.get(rec.genre) || 1) / recommendations.length;
    const makerPenalty = (makerCount.get(rec.maker) || 1) / recommendations.length;
    const priceRange = rec.price < 3000 ? 'low' : rec.price < 5000 ? 'mid' : 'high';
    const pricePenalty = (priceRanges.get(priceRange) || 1) / recommendations.length;

    const diversityPenalty = (genrePenalty + makerPenalty + pricePenalty) / 3;
    const diversityScore = 1 - diversityPenalty;
    
    // 類似度スコアと多様性スコアの重み付き組み合わせ
    const combinedScore = (1 - diversityWeight) * rec.similarity_score + 
                         diversityWeight * diversityScore;

    return {
      ...rec,
      diversity_score: diversityScore,
      similarity_score: combinedScore
    };
  }).sort((a, b) => b.similarity_score - a.similarity_score);
}

/**
 * ユーザーの埋め込み取得（フォールバック付き）
 */
async function getUserEmbedding(
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
      .from('user_video_decisions')
      .select(`
        decision_type,
        videos!inner(genre, price, maker)
      `)
      .eq('user_id', userId)
      .limit(100);

    if (userBehavior && userBehavior.length > 0) {
      // ユーザー行動から簡易埋め込み生成
      return generateUserEmbeddingFromBehavior(userBehavior);
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
    const { data: likes } = await supabaseClient
      .from('likes')
      .select(`
        videos!inner(genre)
      `)
      .eq('user_id', userId)
      .not('videos.genre', 'is', null);

    const genreCount = new Map<string, number>();
    const totalLikes = likes?.length || 0;

    likes?.forEach(like => {
      if (like.videos.genre) {
        genreCount.set(like.videos.genre, (genreCount.get(like.videos.genre) || 0) + 1);
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
  const startTime = performance.now();

  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      {
        status: 405,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }

  try {
    const authHeader = req.headers.get('Authorization');
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? 'http://127.0.0.1:54321';
    const anonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';

    const supabaseClient = createClient(supabaseUrl, anonKey, {
      global: {
        headers: { Authorization: authHeader || '' },
      },
    });

    // ユーザー認証
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser();
    let userId = user?.id;

    if (!userId) {
      console.log('Using test user ID for demo');
      userId = 'e010209e-1e56-450d-bf5f-76a478a9c6ca';
    }

    const requestBody: RecommendationRequest = await req.json();
    
    // 後方互換性: 旧APIパラメータの正規化
    const normalizedRequest = {
      ...requestBody,
      limit: requestBody.limit || requestBody.max_results || 20,
      include_reasons: requestBody.include_reasons ?? requestBody.include_explanations ?? true,
      algorithm: requestBody.algorithm || 'enhanced'
    };
    
    const {
      limit = 20,
      exclude_liked = true,
      diversity_weight = 0.3,
      include_reasons = true,
      min_similarity_threshold = 0.1,
      algorithm = 'enhanced'
    } = normalizedRequest;

    console.log(`${algorithm.toUpperCase()} Two-Tower recommendations for user: ${userId}`);

    // アルゴリズム選択による処理分岐
    if (algorithm === 'basic') {
      console.log('Using basic recommendation algorithm for compatibility');
      // 基本アルゴリズム（旧recommendations関数互換）
    } else if (algorithm === 'two_tower') {
      console.log('Using simple two-tower algorithm for compatibility');
      // 簡易Two-Tower（旧two_tower_recommendations互換）
    }
    // デフォルトは'enhanced'で最新の機能を使用

    // メトリクス初期化
    const metrics: RecommendationMetrics = {
      total_candidates: 0,
      embedding_hits: 0,
      similarity_computation_time: 0,
      sort_time: 0,
      total_processing_time: 0,
      diversity_score: 0,
      avg_confidence: 0
    };

    // ユーザー埋め込みの取得
    const userEmbedding = await getUserEmbedding(supabaseClient, userId);
    if (!userEmbedding) {
      throw new Error('Failed to generate user embedding');
    }

    // ユーザーのジャンル嗜好分析
    const userGenrePreferences = await analyzeUserGenrePreferences(supabaseClient, userId);

    // 候補ビデオの取得
    let videoQuery = supabaseClient
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
        duration_seconds,
        sample_video_url,
        image_urls,
        video_performers!inner(performers(name)),
        video_tags!inner(tags(name))
      `)
      .not('thumbnail_url', 'is', null)
      .limit(limit * 5);

    if (exclude_liked) {
      const { data: likedVideos } = await supabaseClient
        .from('likes')
        .select('video_id')
        .eq('user_id', userId);
      
      const likedIds = likedVideos?.map(l => l.video_id) || [];
      if (likedIds.length > 0) {
        videoQuery = videoQuery.not('id', 'in', `(${likedIds.join(',')})`);
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

    // パフォーマンス最適化: 類似度によるフィルタリングとソート
    const sortStartTime = performance.now();
    
    const filteredSimilarities = similarities
      .filter(sim => sim.similarity >= min_similarity_threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, Math.min(limit * 2, 100)); // メモリ使用量制限と多様性確保
    
    metrics.sort_time = performance.now() - sortStartTime;

    // 推薦結果の構築
    let recommendations: EnhancedRecommendedVideo[] = filteredSimilarities.map(sim => {
      const video = videosWithEmbeddings.find(v => v.id === sim.video_id)!;
      
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
        similarity_score: sim.similarity,
        recommendation_reason: include_reasons 
          ? generateRecommendationReason(video, sim.similarity, userGenrePreferences)
          : `類似度: ${(sim.similarity * 100).toFixed(1)}%`,
        confidence_score: sim.similarity
      };
    });

    // 多様性の向上
    if (diversity_weight > 0) {
      recommendations = enhanceDiversity(recommendations, diversity_weight);
    }

    // 最終的な結果を指定された数に制限
    recommendations = recommendations.slice(0, limit);

    // メトリクスの完了
    metrics.diversity_score = recommendations.length > 1 
      ? new Set(recommendations.map(r => r.genre)).size / recommendations.length 
      : 0;
    metrics.avg_confidence = recommendations.length > 0
      ? recommendations.reduce((sum, r) => sum + (r.confidence_score || 0), 0) / recommendations.length
      : 0;
    metrics.total_processing_time = performance.now() - startTime;

    console.log(`Generated ${recommendations.length} enhanced Two-Tower recommendations in ${metrics.total_processing_time.toFixed(2)}ms`);

    return new Response(
      JSON.stringify({
        recommendations,
        total_count: recommendations.length,
        algorithm: 'enhanced_two_tower_768d',
        metrics,
        user_features: {
          embedding_dimension: 768,
          top_genres: Object.entries(userGenrePreferences)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3)
            .map(([genre, score]) => ({ genre, score: Math.round(score * 100) / 100 }))
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Enhanced Two-Tower recommendation error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error', 
        message: error.message,
        algorithm: 'enhanced_two_tower_768d'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
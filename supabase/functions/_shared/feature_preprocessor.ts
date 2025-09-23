/**
 * 特徴量前処理モジュール
 * 既存Python特徴量処理ロジックのTypeScript移植版
 * TensorFlow.js入力形式への変換機能
 */

import type { ApiResponse } from "./types.ts";

// ============================================================================
// 型定義
// ============================================================================

export interface UserFeatures {
  tags_distribution: Record<string, number>;
  maker_distribution: Record<string, number>;
  performer_distribution: Record<string, number>;
  tag_distribution: Record<string, number>;
  price_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  activity_stats: {
    total_likes: number;
    days_since_first_like: number;
    likes_per_day: number;
    recent_activity: number; // likes in last 7 days
  };
  recency_weight: number[];
}

export interface ItemFeatures {
  basic_features: {
    tags_encoded: number[];
    maker_encoded: number;
    price_normalized: number;
    duration_normalized: number;
  };
  content_features: {
    title_length: number;
    description_length: number;
    tag_count: number;
    performer_count: number;
  };
  metadata_features: {
    release_year: number;
    has_preview: boolean;
    has_sample: boolean;
    image_count: number;
  };
  quality_features: {
    data_completeness: number;
    validation_score: number;
  };
}

export interface TensorFlowJSInput {
  user_id: Float32Array;
  user_features: Float32Array;
}

export interface ItemTensorFlowJSInput {
  item_id: Float32Array;
  item_features: Float32Array;
}

export interface ProcessingStats {
  total_features: number;
  processing_time_ms: number;
  memory_usage_bytes: number;
  feature_dimension: number;
}

// ============================================================================
// 特徴量前処理クラス
// ============================================================================

export class FeaturePreprocessor {
  private readonly USER_FEATURE_DIM = 3; // user_id, tags_features, activity_features
  private readonly ITEM_FEATURE_DIM = 1003; // item_id + comprehensive features
  private readonly MAX_TAGS_CATEGORIES = 50;
  private readonly MAX_MAKER_CATEGORIES = 100;
  private readonly MAX_TAG_CATEGORIES = 200;
  private readonly MAX_PERFORMER_CATEGORIES = 150;

  // エンコーダー用カテゴリマッピング (実際の実装では外部から読み込み)
  private tagsEncoder: Map<string, number> = new Map();
  private makerEncoder: Map<string, number> = new Map();
  private tagEncoder: Map<string, number> = new Map();
  private performerEncoder: Map<string, number> = new Map();

  constructor() {
    this.initializeEncoders();
    console.log("🔧 Feature Preprocessor initialized");
  }

  /**
   * エンコーダー初期化 (実際の実装では学習済みエンコーダーを読み込み)
   */
  private initializeEncoders(): void {
    // サンプルカテゴリーマッピング（実際は学習データから生成）
    const sampleGenres = [
      "アクション", "コメディ", "ドラマ", "ホラー", "SF", "ロマンス",
      "スリラー", "ファンタジー", "アニメ", "ドキュメンタリー"
    ];
    
    sampleGenres.forEach((genre, idx) => {
      this.genreEncoder.set(genre, idx + 1); // 0は未知カテゴリー用
    });

    // その他のエンコーダーも同様に初期化...
    console.log(`📊 Initialized encoders: ${this.genreEncoder.size} genres`);
  }

  /**
   * ユーザー特徴量抽出（既存ロジック移植版）
   */
  extractUserFeatures(likes: any[]): UserFeatures {
    const genreCount: Record<string, number> = {};
    const makerCount: Record<string, number> = {};
    const performerCount: Record<string, number> = {};
    const tagCount: Record<string, number> = {};
    const prices: number[] = [];
    const timestamps: string[] = [];

    // 各いいねから特徴量を抽出
    likes.forEach((like) => {
      const video = like.videos;
      
      // ジャンル分布
      if (video.genre) {
        genreCount[video.genre] = (genreCount[video.genre] || 0) + 1;
      }
      
      // メーカー分布
      if (video.maker) {
        makerCount[video.maker] = (makerCount[video.maker] || 0) + 1;
      }
      
      // 出演者分布
      video.video_performers?.forEach((vp: any) => {
        const performerName = vp.performers?.name;
        if (performerName) {
          performerCount[performerName] = (performerCount[performerName] || 0) + 1;
        }
      });
      
      // タグ分布
      video.video_tags?.forEach((vt: any) => {
        const tagName = vt.tags?.name;
        if (tagName) {
          tagCount[tagName] = (tagCount[tagName] || 0) + 1;
        }
      });
      
      // 価格
      if (video.price) {
        prices.push(video.price);
      }
      
      timestamps.push(like.created_at);
    });

    // 正規化
    const totalLikes = likes.length;
    const normalizeDistribution = (counts: Record<string, number>) => {
      const normalized: Record<string, number> = {};
      Object.entries(counts).forEach(([key, count]) => {
        normalized[key] = count / totalLikes;
      });
      return normalized;
    };

    // 価格統計
    const priceMean = prices.reduce((a, b) => a + b, 0) / prices.length || 0;
    const priceVariance = prices.reduce((acc, price) => acc + Math.pow(price - priceMean, 2), 0) / prices.length || 0;
    const priceStd = Math.sqrt(priceVariance);

    // アクティビティ統計
    const now = new Date();
    const firstLike = new Date(timestamps[timestamps.length - 1]);
    const daysSinceFirst = Math.max(1, (now.getTime() - firstLike.getTime()) / (1000 * 60 * 60 * 24));
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const recentActivity = timestamps.filter(ts => new Date(ts) > sevenDaysAgo).length;

    // 時系列重み（新しいいいねほど重要）
    const recencyWeights = timestamps.map((_, index) => {
      return Math.exp(-0.1 * index); // 指数減衰
    });

    return {
      genre_distribution: normalizeDistribution(genreCount),
      maker_distribution: normalizeDistribution(makerCount),
      performer_distribution: normalizeDistribution(performerCount),
      tag_distribution: normalizeDistribution(tagCount),
      price_stats: {
        mean: priceMean,
        std: priceStd,
        min: Math.min(...prices) || 0,
        max: Math.max(...prices) || 0,
      },
      activity_stats: {
        total_likes: totalLikes,
        days_since_first_like: daysSinceFirst,
        likes_per_day: totalLikes / daysSinceFirst,
        recent_activity: recentActivity,
      },
      recency_weight: recencyWeights,
    };
  }

  /**
   * タグエンコーディング（tags-basedシステム用）
   */
  extractTagsEncoding(video: any): number[] {
    const encoding = new Array(10).fill(0); // 10次元のタグエンコーディング

    if (video.video_tags && Array.isArray(video.video_tags)) {
      const tags = video.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);

      // ジャンルタグを抽出
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
      if (genreTag) {
        const genreIndex = this.genreEncoder.get(genreTag) || 0;
        encoding[0] = genreIndex / this.MAX_GENRE_CATEGORIES;
      }

      // その他のタグをエンコード
      tags.slice(0, 9).forEach((tag: string, index: number) => {
        if (index < 9) {
          const tagHash = this.hashString(tag) % 1000;
          encoding[index + 1] = tagHash / 1000;
        }
      });
    }

    return encoding;
  }

  /**
   * 文字列ハッシュ関数
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * アイテム特徴量抽出（DMM APIデータから）
   */
  extractItemFeatures(video: any): ItemFeatures {
    // 基本特徴量
    const basic_features = {
      tags_encoded: this.extractTagsEncoding(video),
      maker_encoded: this.makerEncoder.get(video.maker) || 0,
      price_normalized: video.price ? Math.log(video.price + 1) / Math.log(10000) : 0, // log正規化
      duration_normalized: video.duration_seconds ? video.duration_seconds / 7200 : 0, // 2時間で正規化
    };

    // コンテンツ特徴量
    const content_features = {
      title_length: video.title ? video.title.length : 0,
      description_length: video.description ? video.description.length : 0,
      tag_count: video.video_tags ? video.video_tags.length : 0,
      performer_count: video.video_performers ? video.video_performers.length : 0,
    };

    // メタデータ特徴量
    const releaseDate = video.created_at ? new Date(video.created_at) : new Date();
    const metadata_features = {
      release_year: releaseDate.getFullYear() - 2000, // 2000年基準
      has_preview: !!video.preview_video_url,
      has_sample: !!video.sample_video_url,
      image_count: video.image_urls ? video.image_urls.length : 0,
    };

    // 品質特徴量
    const completeness = [
      video.title, video.description, video.genre, video.maker,
      video.price, video.thumbnail_url
    ].filter(Boolean).length / 6; // 必須フィールド完成度

    const quality_features = {
      data_completeness: completeness,
      validation_score: video.validation_status === "valid" ? 1.0 : 0.0,
    };

    return {
      basic_features,
      content_features,
      metadata_features,
      quality_features,
    };
  }

  /**
   * ユーザー特徴量をTensorFlow.js入力形式に変換
   */
  prepareUserTensorInput(userId: number, features: UserFeatures): TensorFlowJSInput {
    const startTime = performance.now();

    // ユーザーID (1次元)
    const user_id = new Float32Array([userId]);

    // 特徴量ベクトル構築 (3次元: genre, activity, price)
    const user_features = new Float32Array(this.USER_FEATURE_DIM);
    
    // ジャンル特徴量 (主要ジャンルの重み付き平均)
    let genreScore = 0;
    let genreWeight = 0;
    Object.entries(features.genre_distribution).forEach(([genre, weight]) => {
      const encoded = this.genreEncoder.get(genre) || 0;
      genreScore += encoded * weight;
      genreWeight += weight;
    });
    user_features[0] = genreWeight > 0 ? genreScore / genreWeight / this.MAX_GENRE_CATEGORIES : 0;

    // アクティビティ特徴量
    user_features[1] = Math.tanh(features.activity_stats.likes_per_day / 10); // tanh正規化

    // 価格特徴量
    user_features[2] = features.price_stats.mean > 0 
      ? Math.log(features.price_stats.mean) / Math.log(10000) 
      : 0;

    const processingTime = performance.now() - startTime;
    
    console.log(`🔄 User features prepared: ${user_features.length}D in ${processingTime.toFixed(2)}ms`);

    return {
      user_id,
      user_features,
    };
  }

  /**
   * アイテム特徴量をTensorFlow.js入力形式に変換
   */
  prepareItemTensorInput(itemId: number, features: ItemFeatures): ItemTensorFlowJSInput {
    const startTime = performance.now();

    // アイテムID (1次元)
    const item_id = new Float32Array([itemId]);

    // 包括的特徴量ベクトル構築 (1003次元)
    const item_features = new Float32Array(this.ITEM_FEATURE_DIM);
    
    let idx = 0;

    // 基本特徴量 (4次元)
    // Tags encoding (複数のタグを処理)
    const tagsEncoded = features.basic_features.tags_encoded;
    for (let i = 0; i < Math.min(tagsEncoded.length, 10); i++) {
      item_features[idx++] = tagsEncoded[i];
    }
    item_features[idx++] = features.basic_features.maker_encoded / this.MAX_MAKER_CATEGORIES;
    item_features[idx++] = features.basic_features.price_normalized;
    item_features[idx++] = features.basic_features.duration_normalized;

    // コンテンツ特徴量 (4次元)
    item_features[idx++] = Math.tanh(features.content_features.title_length / 100);
    item_features[idx++] = Math.tanh(features.content_features.description_length / 1000);
    item_features[idx++] = Math.tanh(features.content_features.tag_count / 20);
    item_features[idx++] = Math.tanh(features.content_features.performer_count / 10);

    // メタデータ特徴量 (4次元)
    item_features[idx++] = features.metadata_features.release_year / 25; // 25年レンジで正規化
    item_features[idx++] = features.metadata_features.has_preview ? 1.0 : 0.0;
    item_features[idx++] = features.metadata_features.has_sample ? 1.0 : 0.0;
    item_features[idx++] = Math.tanh(features.metadata_features.image_count / 10);

    // 品質特徴量 (2次元)
    item_features[idx++] = features.quality_features.data_completeness;
    item_features[idx++] = features.quality_features.validation_score;

    // 残り次元をゼロ埋め（実際の実装では更に詳細な特徴量を追加）
    while (idx < this.ITEM_FEATURE_DIM) {
      item_features[idx++] = 0.0;
    }

    const processingTime = performance.now() - startTime;
    
    console.log(`🔄 Item features prepared: ${item_features.length}D in ${processingTime.toFixed(2)}ms`);

    return {
      item_id,
      item_features,
    };
  }

  /**
   * バッチユーザー特徴量処理
   */
  prepareBatchUserInputs(
    userIds: number[], 
    userLikes: any[][]
  ): { inputs: TensorFlowJSInput[], stats: ProcessingStats } {
    const startTime = performance.now();
    const inputs: TensorFlowJSInput[] = [];

    for (let i = 0; i < userIds.length; i++) {
      const features = this.extractUserFeatures(userLikes[i]);
      const tensorInput = this.prepareUserTensorInput(userIds[i], features);
      inputs.push(tensorInput);
    }

    const processingTime = performance.now() - startTime;
    const memoryUsage = inputs.length * (this.USER_FEATURE_DIM + 1) * 4; // Float32 = 4 bytes

    return {
      inputs,
      stats: {
        total_features: inputs.length,
        processing_time_ms: processingTime,
        memory_usage_bytes: memoryUsage,
        feature_dimension: this.USER_FEATURE_DIM,
      },
    };
  }

  /**
   * バッチアイテム特徴量処理
   */
  prepareBatchItemInputs(
    itemIds: number[], 
    videos: any[]
  ): { inputs: ItemTensorFlowJSInput[], stats: ProcessingStats } {
    const startTime = performance.now();
    const inputs: ItemTensorFlowJSInput[] = [];

    for (let i = 0; i < itemIds.length; i++) {
      const features = this.extractItemFeatures(videos[i]);
      const tensorInput = this.prepareItemTensorInput(itemIds[i], features);
      inputs.push(tensorInput);
    }

    const processingTime = performance.now() - startTime;
    const memoryUsage = inputs.length * (this.ITEM_FEATURE_DIM + 1) * 4; // Float32 = 4 bytes

    return {
      inputs,
      stats: {
        total_features: inputs.length,
        processing_time_ms: processingTime,
        memory_usage_bytes: memoryUsage,
        feature_dimension: this.ITEM_FEATURE_DIM,
      },
    };
  }

  /**
   * 特徴量統計情報取得
   */
  getFeatureStats(): ApiResponse<{
    user_feature_dimension: number;
    item_feature_dimension: number;
    encoder_stats: {
      genres: number;
      makers: number;
      tags: number;
      performers: number;
    };
  }> {
    return {
      success: true,
      data: {
        user_feature_dimension: this.USER_FEATURE_DIM,
        item_feature_dimension: this.ITEM_FEATURE_DIM,
        encoder_stats: {
          genres: this.genreEncoder.size,
          makers: this.makerEncoder.size,
          tags: this.tagEncoder.size,
          performers: this.performerEncoder.size,
        },
      },
    };
  }

  /**
   * 特徴量互換性検証
   */
  validateFeatureCompatibility(pythonFeatures: any): ApiResponse<{
    compatible: boolean;
    differences: string[];
  }> {
    const differences: string[] = [];

    // Python実装との互換性チェック
    const requiredFields = [
      'genre_distribution', 'maker_distribution', 'performer_distribution',
      'tag_distribution', 'price_stats', 'activity_stats', 'recency_weight'
    ];

    requiredFields.forEach(field => {
      if (!pythonFeatures[field]) {
        differences.push(`Missing field: ${field}`);
      }
    });

    const compatible = differences.length === 0;

    return {
      success: true,
      data: {
        compatible,
        differences,
      },
    };
  }
}

// ============================================================================
// グローバルプリプロセッサーインスタンス (Singleton)
// ============================================================================

let globalPreprocessor: FeaturePreprocessor | null = null;

/**
 * グローバル特徴量プリプロセッサー取得
 */
export function getFeaturePreprocessor(): FeaturePreprocessor {
  if (!globalPreprocessor) {
    globalPreprocessor = new FeaturePreprocessor();
  }
  return globalPreprocessor;
}

/**
 * 特徴量プリプロセッサー初期化 (Edge Functions用)
 */
export function initializeFeaturePreprocessor(): FeaturePreprocessor {
  globalPreprocessor = new FeaturePreprocessor();
  return globalPreprocessor;
}
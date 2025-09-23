/**
 * TensorFlow.jsモデルローダー
 * Supabase StorageからTwo-Towerモデルを読み込み・キャッシュ
 */

import * as tf from "npm:@tensorflow/tfjs@4.20.0";
import { createClient } from "npm:@supabase/supabase-js@2.38.0";
import type { ApiResponse } from "./types.ts";

// ============================================================================
// 型定義
// ============================================================================

export interface ModelConfig {
  name: string;
  version?: string;
  storage_path: string;
  input_shape: number[][];
  output_shape: number[];
  parameters: number;
  layers: number;
}

export interface ModelLoadResult {
  model: tf.LayersModel;
  config: ModelConfig;
  loaded_at: string;
  cache_key: string;
}

export interface InferenceResult {
  embeddings: Float32Array;
  shape: number[];
  processing_time_ms: number;
  model_used: string;
}

export interface ModelCache {
  user_tower?: ModelLoadResult;
  item_tower?: ModelLoadResult;
  cache_timestamp: number;
  cache_ttl: number; // seconds
}

// ============================================================================
// モデルローダークラス
// ============================================================================

export class TensorFlowJSModelLoader {
  private supabase: any;
  private cache: ModelCache;
  private readonly CACHE_TTL = 3600; // 1 hour
  private readonly BUCKET_NAME = "ml-models";
  private readonly MAX_MEMORY_MB = 128; // Edge Functions memory limit

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    // Supabase client初期化
    this.supabase = createClient(
      supabaseUrl || Deno.env.get("SUPABASE_URL")!,
      supabaseKey || Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    // キャッシュ初期化
    this.cache = {
      cache_timestamp: 0,
      cache_ttl: this.CACHE_TTL,
    };

    console.log("🚀 TensorFlow.js Model Loader initialized");
  }

  /**
   * モデル設定をStorageから読み込み
   */
  private async loadModelConfig(modelName: string, version?: string): Promise<ModelConfig> {
    try {
      const versionPath = version || "current_version.json";
      const configPath = version 
        ? `${modelName}/v${version}/model_metadata.json`
        : `${modelName}/${versionPath}`;

      // バージョン情報取得
      let targetVersion = version;
      if (!version) {
        const { data: versionData, error: versionError } = await this.supabase.storage
          .from(this.BUCKET_NAME)
          .download(`${modelName}/current_version.json`);

        if (versionError) throw versionError;

        const versionText = await versionData.text();
        const versionInfo = JSON.parse(versionText);
        targetVersion = versionInfo.current_version;
      }

      // モデルメタデータ取得
      const metadataPath = `${modelName}/v${targetVersion}/model_metadata.json`;
      const { data: metadataData, error: metadataError } = await this.supabase.storage
        .from(this.BUCKET_NAME)
        .download(metadataPath);

      if (metadataError) throw metadataError;

      const metadataText = await metadataData.text();
      const metadata = JSON.parse(metadataText);

      return {
        name: modelName,
        version: targetVersion,
        storage_path: `${modelName}/v${targetVersion}`,
        input_shape: metadata.input_shape,
        output_shape: metadata.output_shape,
        parameters: metadata.parameters,
        layers: metadata.layers,
      };

    } catch (error) {
      console.error(`❌ Failed to load model config for ${modelName}:`, error);
      throw new Error(`Model config loading failed: ${error.message}`);
    }
  }

  /**
   * TensorFlow.jsモデルをStorageから読み込み
   */
  private async loadModelFromStorage(config: ModelConfig): Promise<tf.LayersModel> {
    try {
      console.log(`📥 Loading TensorFlow.js model: ${config.name} v${config.version}`);

      // model.json取得
      const modelJsonPath = `${config.storage_path}/model.json`;
      const { data: modelJsonData, error: jsonError } = await this.supabase.storage
        .from(this.BUCKET_NAME)
        .download(modelJsonPath);

      if (jsonError) throw jsonError;

      const modelJsonText = await modelJsonData.text();
      const modelJson = JSON.parse(modelJsonText);

      // weights.binファイルのURL生成
      const weightsPath = `${config.storage_path}/${modelJson.weightsManifest[0].paths[0]}`;
      const { data: { publicUrl } } = this.supabase.storage
        .from(this.BUCKET_NAME)
        .getPublicUrl(weightsPath);

      // モデル読み込み用のfetch関数作成
      const fetchFunc = async (url: string) => {
        if (url.endsWith('.bin')) {
          // weights file
          const { data: weightsData, error: weightsError } = await this.supabase.storage
            .from(this.BUCKET_NAME)
            .download(weightsPath);

          if (weightsError) throw weightsError;

          const arrayBuffer = await weightsData.arrayBuffer();
          return new Response(arrayBuffer);
        } else {
          // model.json file
          return new Response(modelJsonText, {
            headers: { 'Content-Type': 'application/json' }
          });
        }
      };

      // TensorFlow.jsモデル読み込み
      const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson, new Uint8Array(await (await this.supabase.storage
        .from(this.BUCKET_NAME)
        .download(weightsPath)).data.arrayBuffer())));

      console.log(`✅ Model loaded successfully: ${config.name} (${config.parameters} parameters)`);
      return model;

    } catch (error) {
      console.error(`❌ Failed to load model ${config.name}:`, error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }

  /**
   * キャッシュ有効性確認
   */
  private isCacheValid(): boolean {
    const now = Date.now() / 1000;
    return (now - this.cache.cache_timestamp) < this.cache.cache_ttl;
  }

  /**
   * メモリ使用量チェック
   */
  private checkMemoryUsage(): void {
    const memInfo = tf.memory();
    const usedMB = memInfo.numBytes / (1024 * 1024);
    
    if (usedMB > this.MAX_MEMORY_MB * 0.8) {
      console.warn(`⚠️ High memory usage: ${usedMB.toFixed(2)}MB / ${this.MAX_MEMORY_MB}MB`);
      
      // メモリクリーンアップ
      tf.disposeVariables();
      
      if (usedMB > this.MAX_MEMORY_MB * 0.9) {
        // キャッシュクリア
        this.clearCache();
      }
    }
  }

  /**
   * キャッシュクリア
   */
  private clearCache(): void {
    if (this.cache.user_tower?.model) {
      this.cache.user_tower.model.dispose();
    }
    if (this.cache.item_tower?.model) {
      this.cache.item_tower.model.dispose();
    }
    
    this.cache = {
      cache_timestamp: 0,
      cache_ttl: this.CACHE_TTL,
    };
    
    console.log("🧹 Model cache cleared");
  }

  /**
   * User Towerモデル読み込み
   */
  async loadUserTower(version?: string): Promise<ModelLoadResult> {
    try {
      // キャッシュ確認
      if (this.cache.user_tower && this.isCacheValid()) {
        console.log("📋 Using cached user tower model");
        return this.cache.user_tower;
      }

      // メモリチェック
      this.checkMemoryUsage();

      // モデル設定読み込み
      const config = await this.loadModelConfig("user_tower", version);
      
      // モデル読み込み
      const model = await this.loadModelFromStorage(config);

      const result: ModelLoadResult = {
        model,
        config,
        loaded_at: new Date().toISOString(),
        cache_key: `user_tower_v${config.version}`,
      };

      // キャッシュ更新
      this.cache.user_tower = result;
      this.cache.cache_timestamp = Date.now() / 1000;

      return result;

    } catch (error) {
      console.error("❌ Failed to load user tower:", error);
      throw error;
    }
  }

  /**
   * Item Towerモデル読み込み
   */
  async loadItemTower(version?: string): Promise<ModelLoadResult> {
    try {
      // キャッシュ確認
      if (this.cache.item_tower && this.isCacheValid()) {
        console.log("📋 Using cached item tower model");
        return this.cache.item_tower;
      }

      // メモリチェック
      this.checkMemoryUsage();

      // モデル設定読み込み
      const config = await this.loadModelConfig("item_tower", version);
      
      // モデル読み込み
      const model = await this.loadModelFromStorage(config);

      const result: ModelLoadResult = {
        model,
        config,
        loaded_at: new Date().toISOString(),
        cache_key: `item_tower_v${config.version}`,
      };

      // キャッシュ更新
      this.cache.item_tower = result;
      this.cache.cache_timestamp = Date.now() / 1000;

      return result;

    } catch (error) {
      console.error("❌ Failed to load item tower:", error);
      throw error;
    }
  }

  /**
   * 両タワー並列読み込み
   */
  async loadBothTowers(version?: string): Promise<{
    user_tower: ModelLoadResult;
    item_tower: ModelLoadResult;
  }> {
    try {
      console.log("🚀 Loading both towers in parallel...");
      
      const [userTower, itemTower] = await Promise.all([
        this.loadUserTower(version),
        this.loadItemTower(version),
      ]);

      console.log("✅ Both towers loaded successfully");
      
      return {
        user_tower: userTower,
        item_tower: itemTower,
      };

    } catch (error) {
      console.error("❌ Failed to load both towers:", error);
      throw error;
    }
  }

  /**
   * User Tower推論実行
   */
  async predictUserTower(
    userFeatures: number[],
    userId: number,
    version?: string
  ): Promise<InferenceResult> {
    const startTime = performance.now();
    
    try {
      const userTower = await this.loadUserTower(version);
      
      // 入力データ準備
      const userIdTensor = tf.tensor1d([userId], 'float32');
      const userFeaturesTensor = tf.tensor2d([userFeatures], [1, userFeatures.length], 'float32');
      
      // 推論実行
      const prediction = userTower.model.predict([userIdTensor, userFeaturesTensor]) as tf.Tensor;
      
      // 結果取得
      const embeddings = await prediction.data() as Float32Array;
      const shape = prediction.shape;
      
      // テンソル解放
      userIdTensor.dispose();
      userFeaturesTensor.dispose();
      prediction.dispose();
      
      const processingTime = performance.now() - startTime;
      
      return {
        embeddings,
        shape,
        processing_time_ms: processingTime,
        model_used: userTower.cache_key,
      };

    } catch (error) {
      console.error("❌ User tower prediction failed:", error);
      throw error;
    }
  }

  /**
   * Item Tower推論実行
   */
  async predictItemTower(
    itemId: number,
    itemFeatures: number[],
    version?: string
  ): Promise<InferenceResult> {
    const startTime = performance.now();
    
    try {
      const itemTower = await this.loadItemTower(version);
      
      // 入力データ準備
      const itemIdTensor = tf.tensor1d([itemId], 'float32');
      const itemFeaturesTensor = tf.tensor2d([itemFeatures], [1, itemFeatures.length], 'float32');
      
      // 推論実行
      const prediction = itemTower.model.predict([itemIdTensor, itemFeaturesTensor]) as tf.Tensor;
      
      // 結果取得
      const embeddings = await prediction.data() as Float32Array;
      const shape = prediction.shape;
      
      // テンソル解放
      itemIdTensor.dispose();
      itemFeaturesTensor.dispose();
      prediction.dispose();
      
      const processingTime = performance.now() - startTime;
      
      return {
        embeddings,
        shape,
        processing_time_ms: processingTime,
        model_used: itemTower.cache_key,
      };

    } catch (error) {
      console.error("❌ Item tower prediction failed:", error);
      throw error;
    }
  }

  /**
   * モデル情報取得
   */
  getModelInfo(): ApiResponse<{
    cache_status: {
      user_tower_cached: boolean;
      item_tower_cached: boolean;
      cache_age_seconds: number;
      cache_valid: boolean;
    };
    memory_usage: {
      num_tensors: number;
      num_bytes: number;
      num_bytes_mb: number;
    };
  }> {
    try {
      const memInfo = tf.memory();
      const cacheAge = Date.now() / 1000 - this.cache.cache_timestamp;
      
      return {
        success: true,
        data: {
          cache_status: {
            user_tower_cached: !!this.cache.user_tower,
            item_tower_cached: !!this.cache.item_tower,
            cache_age_seconds: cacheAge,
            cache_valid: this.isCacheValid(),
          },
          memory_usage: {
            num_tensors: memInfo.numTensors,
            num_bytes: memInfo.numBytes,
            num_bytes_mb: memInfo.numBytes / (1024 * 1024),
          },
        },
      };

    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * リソース解放
   */
  dispose(): void {
    this.clearCache();
    console.log("🗑️ TensorFlow.js Model Loader disposed");
  }
}

// ============================================================================
// グローバルローダーインスタンス (Singleton)
// ============================================================================

let globalModelLoader: TensorFlowJSModelLoader | null = null;

/**
 * グローバルモデルローダー取得
 */
export function getModelLoader(): TensorFlowJSModelLoader {
  if (!globalModelLoader) {
    globalModelLoader = new TensorFlowJSModelLoader();
  }
  return globalModelLoader;
}

/**
 * モデルローダー初期化 (Edge Functions用)
 */
export function initializeModelLoader(supabaseUrl?: string, supabaseKey?: string): TensorFlowJSModelLoader {
  if (globalModelLoader) {
    globalModelLoader.dispose();
  }
  
  globalModelLoader = new TensorFlowJSModelLoader(supabaseUrl, supabaseKey);
  return globalModelLoader;
}
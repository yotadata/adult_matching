/**
 * エンベディング無効化と再生成スクリプト
 * tags-basedシステム移行に伴う既存エンベディングの更新
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

interface EmbeddingRegenerationConfig {
  batchSize: number;
  delayBetweenBatches: number; // milliseconds
  maxRetries: number;
}

const config: EmbeddingRegenerationConfig = {
  batchSize: 50,
  delayBetweenBatches: 1000,
  maxRetries: 3
};

class EmbeddingRegenerator {
  private supabaseClient: any;

  constructor() {
    const supabaseUrl = Deno.env.get('SUPABASE_URL') || 'http://127.0.0.1:54321';
    const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('SUPABASE_ANON_KEY');

    if (!serviceKey) {
      throw new Error('SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY is required');
    }

    this.supabaseClient = createClient(supabaseUrl, serviceKey);
  }

  /**
   * メイン実行関数
   */
  async run(): Promise<void> {
    console.log('🚀 Embedding regeneration started');
    console.log(`Configuration: batch_size=${config.batchSize}, delay=${config.delayBetweenBatches}ms`);

    try {
      // Step 1: 既存エンベディングの無効化
      await this.invalidateExistingEmbeddings();

      // Step 2: ユーザーエンベディングの再生成
      await this.regenerateUserEmbeddings();

      // Step 3: ビデオエンベディングの確認
      await this.checkVideoEmbeddings();

      // Step 4: 統計レポート
      await this.generateReport();

      console.log('✅ Embedding regeneration completed successfully');

    } catch (error) {
      console.error('❌ Embedding regeneration failed:', error);
      throw error;
    }
  }

  /**
   * 既存エンベディングの無効化
   */
  private async invalidateExistingEmbeddings(): Promise<void> {
    console.log('\n📝 Step 1: Invalidating existing embeddings...');

    // 古いgenre-basedエンベディングをマーク
    const { error: markError } = await this.supabaseClient
      .from('user_embeddings')
      .update({
        is_valid: false,
        invalidated_at: new Date().toISOString(),
        invalidation_reason: 'genre_to_tags_migration'
      })
      .lt('updated_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()); // 24時間以上前のもの

    if (markError) {
      console.error('Error marking old embeddings:', markError);
      throw markError;
    }

    // 統計取得
    const { count, error: countError } = await this.supabaseClient
      .from('user_embeddings')
      .select('*', { count: 'exact', head: true })
      .eq('is_valid', false);

    if (!countError) {
      console.log(`✅ Marked ${count || 0} existing embeddings as invalid`);
    }
  }

  /**
   * ユーザーエンベディングの再生成
   */
  private async regenerateUserEmbeddings(): Promise<void> {
    console.log('\n🔄 Step 2: Regenerating user embeddings...');

    // アクティブユーザーのリストを取得
    const { data: users, error: usersError } = await this.supabaseClient
      .from('likes')
      .select('user_id')
      .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()) // 過去30日
      .order('created_at', { ascending: false });

    if (usersError) {
      throw usersError;
    }

    const uniqueUsers = [...new Set(users?.map(u => u.user_id) || [])];
    console.log(`Found ${uniqueUsers.length} active users for embedding regeneration`);

    let processed = 0;
    let errors = 0;

    // バッチ処理
    for (let i = 0; i < uniqueUsers.length; i += config.batchSize) {
      const batch = uniqueUsers.slice(i, i + config.batchSize);

      console.log(`Processing batch ${Math.floor(i / config.batchSize) + 1}/${Math.ceil(uniqueUsers.length / config.batchSize)}`);

      const batchPromises = batch.map(userId => this.regenerateUserEmbedding(userId));
      const results = await Promise.allSettled(batchPromises);

      results.forEach(result => {
        if (result.status === 'fulfilled') {
          processed++;
        } else {
          errors++;
          console.error('User embedding regeneration error:', result.reason);
        }
      });

      // バッチ間の遅延
      if (i + config.batchSize < uniqueUsers.length) {
        await new Promise(resolve => setTimeout(resolve, config.delayBetweenBatches));
      }
    }

    console.log(`✅ User embedding regeneration completed: ${processed} successful, ${errors} errors`);
  }

  /**
   * 個別ユーザーエンベディングの再生成
   */
  private async regenerateUserEmbedding(userId: string): Promise<void> {
    try {
      // update_user_embedding Edge Functionを呼び出し
      const response = await fetch(`${Deno.env.get('SUPABASE_URL')}/functions/v1/update_user_embedding`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')}`
        },
        body: JSON.stringify({
          user_id: userId,
          force_update: true,
          batch_phase: 'regeneration'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      console.log(`✅ Regenerated embedding for user ${userId}`);

    } catch (error) {
      console.error(`❌ Failed to regenerate embedding for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * ビデオエンベディングの確認
   */
  private async checkVideoEmbeddings(): Promise<void> {
    console.log('\n🎬 Step 3: Checking video embeddings...');

    const { count: totalVideos, error: videoCountError } = await this.supabaseClient
      .from('videos')
      .select('*', { count: 'exact', head: true });

    const { count: embeddedVideos, error: embeddingCountError } = await this.supabaseClient
      .from('video_embeddings')
      .select('*', { count: 'exact', head: true });

    if (!videoCountError && !embeddingCountError) {
      console.log(`📊 Video embedding coverage: ${embeddedVideos}/${totalVideos} (${((embeddedVideos / totalVideos) * 100).toFixed(1)}%)`);

      if (embeddedVideos < totalVideos * 0.9) {
        console.log('⚠️ Warning: Video embedding coverage is below 90%');
        console.log('Consider running item_embedding_generator to improve coverage');
      } else {
        console.log('✅ Video embedding coverage is adequate');
      }
    }
  }

  /**
   * 統計レポートの生成
   */
  private async generateReport(): Promise<void> {
    console.log('\n📊 Step 4: Generating final report...');

    const queries = [
      // 有効なユーザーエンベディング数
      this.supabaseClient
        .from('user_embeddings')
        .select('*', { count: 'exact', head: true })
        .eq('is_valid', true),

      // 無効化されたエンベディング数
      this.supabaseClient
        .from('user_embeddings')
        .select('*', { count: 'exact', head: true })
        .eq('is_valid', false),

      // 最近更新されたエンベディング数（24時間以内）
      this.supabaseClient
        .from('user_embeddings')
        .select('*', { count: 'exact', head: true })
        .gte('updated_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString())
    ];

    const [validResult, invalidResult, recentResult] = await Promise.all(queries);

    console.log('\n📈 Final Statistics:');
    console.log(`Valid user embeddings: ${validResult.count || 0}`);
    console.log(`Invalidated embeddings: ${invalidResult.count || 0}`);
    console.log(`Recently updated (24h): ${recentResult.count || 0}`);

    const totalEmbeddings = (validResult.count || 0) + (invalidResult.count || 0);
    if (totalEmbeddings > 0) {
      const validPercentage = ((validResult.count || 0) / totalEmbeddings * 100).toFixed(1);
      console.log(`Validity rate: ${validPercentage}%`);
    }

    // 完了状況をファイルに保存
    const reportData = {
      timestamp: new Date().toISOString(),
      migration_phase: 'tags_based_system',
      statistics: {
        valid_embeddings: validResult.count || 0,
        invalidated_embeddings: invalidResult.count || 0,
        recent_updates: recentResult.count || 0,
        validity_rate: totalEmbeddings > 0 ? (validResult.count || 0) / totalEmbeddings : 0
      },
      config: config
    };

    try {
      await Deno.writeTextFile(
        '/tmp/embedding_regeneration_report.json',
        JSON.stringify(reportData, null, 2)
      );
      console.log('📄 Report saved to /tmp/embedding_regeneration_report.json');
    } catch (writeError) {
      console.log('⚠️ Could not save report file, but regeneration completed successfully');
    }
  }
}

// メイン実行
if (import.meta.main) {
  const regenerator = new EmbeddingRegenerator();

  try {
    await regenerator.run();
    Deno.exit(0);
  } catch (error) {
    console.error('Script failed:', error);
    Deno.exit(1);
  }
}
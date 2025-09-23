const { createClient } = require('@supabase/supabase-js');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// Supabase設定
const supabaseUrl = 'http://127.0.0.1:54321';
const serviceRoleKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, serviceRoleKey);

// DMM API設定
const DMM_API_CONFIG = {
  api_id: 'W63Kd4A4ym2DaycFcXSU',
  affiliate_id: 'yotadata2-990',
  baseUrl: 'https://api.dmm.com/affiliate/v3/ItemList'
};

// 進捗管理設定
const PROGRESS_FILE = path.join(__dirname, 'dmm_bulk_sync_progress.json');
const CACHE_FILE = path.join(__dirname, 'dmm_existing_ids_cache.json');

// パフォーマンス設定
const CONFIG = {
  batchSize: 100,           // 1回あたりの取得件数
  maxConcurrentPages: 1,    // 同時処理ページ数（API制限のため1）
  rateLimitDelay: 800,      // APIコール間隔（ミリ秒）
  progressSaveInterval: 10, // 進捗保存間隔（ページ数）
  cacheRefreshInterval: 100,// キャッシュリフレッシュ間隔
  maxRetries: 3,           // エラー時の最大再試行回数
  targetDateStart: '2006-02-14', // レビューデータ最古日付
  estimatedTotalPages: 500  // 推定総ページ数
};

class EfficientDmmBulkSync {
  constructor() {
    this.existingIds = new Set();
    this.stats = {
      totalInserted: 0,
      totalSkipped: 0,
      totalErrors: 0,
      pagesProcessed: 0,
      startTime: Date.now(),
      lastSaveTime: Date.now()
    };
    this.progress = null;
  }

  async initialize() {
    console.log('🚀 Efficient DMM Bulk Sync - Initialize');
    console.log('=====================================');
    
    // 進捗ファイル読み込み
    await this.loadProgress();
    
    // 既存IDキャッシュ読み込み/作成
    await this.loadOrCreateIdCache();
    
    console.log(`📊 Initialization Complete:`);
    console.log(`   🗂️  Cached existing IDs: ${this.existingIds.size.toLocaleString()}`);
    console.log(`   📄 Resume from page: ${this.progress.currentPage}`);
    console.log(`   📈 Progress: ${((this.progress.currentPage / CONFIG.estimatedTotalPages) * 100).toFixed(1)}%`);
  }

  async loadProgress() {
    try {
      const progressData = await fs.readFile(PROGRESS_FILE, 'utf8');
      this.progress = JSON.parse(progressData);
      console.log(`✅ Loaded existing progress: page ${this.progress.currentPage}`);
    } catch (error) {
      // 新規開始
      this.progress = {
        currentPage: 1,
        totalPages: CONFIG.estimatedTotalPages,
        lastProcessedAt: new Date().toISOString(),
        completed: false
      };
      console.log('🆕 Starting new bulk sync session');
    }
  }

  async saveProgress() {
    this.progress.lastProcessedAt = new Date().toISOString();
    await fs.writeFile(PROGRESS_FILE, JSON.stringify(this.progress, null, 2));
    this.stats.lastSaveTime = Date.now();
  }

  async loadOrCreateIdCache() {
    try {
      // キャッシュファイルから読み込み
      const cacheData = await fs.readFile(CACHE_FILE, 'utf8');
      const cachedIds = JSON.parse(cacheData);
      this.existingIds = new Set(cachedIds);
      console.log(`✅ Loaded ID cache: ${this.existingIds.size.toLocaleString()} existing IDs`);
    } catch (error) {
      // データベースから既存IDを取得してキャッシュ作成
      console.log('🔄 Creating ID cache from database...');
      await this.refreshIdCache();
    }
  }

  async refreshIdCache() {
    try {
      const { data: videos, error } = await supabase
        .from('videos')
        .select('external_id')
        .eq('source', 'dmm');

      if (error) throw error;

      this.existingIds = new Set(videos.map(v => v.external_id));
      
      // キャッシュファイル保存
      await fs.writeFile(CACHE_FILE, JSON.stringify([...this.existingIds], null, 2));
      
      console.log(`🔄 Refreshed ID cache: ${this.existingIds.size.toLocaleString()} existing IDs`);
    } catch (error) {
      console.error('❌ Error refreshing ID cache:', error.message);
      throw error;
    }
  }

  async fetchDmmApiData(page, retryCount = 0) {
    const params = {
      api_id: DMM_API_CONFIG.api_id,
      affiliate_id: DMM_API_CONFIG.affiliate_id,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      hits: CONFIG.batchSize,
      offset: ((page - 1) * CONFIG.batchSize) + 1,
      sort: 'date',
      output: 'json'
    };

    try {
      const response = await axios.get(DMM_API_CONFIG.baseUrl, { 
        params: params,
        timeout: 30000
      });

      if (response.data?.result?.items) {
        return {
          items: response.data.result.items,
          totalCount: response.data.result.total_count || 0
        };
      } else {
        throw new Error('Invalid API response format');
      }
    } catch (error) {
      if (retryCount < CONFIG.maxRetries) {
        console.log(`⚠️  API Error (retry ${retryCount + 1}/${CONFIG.maxRetries}):`, error.message);
        await new Promise(resolve => setTimeout(resolve, (retryCount + 1) * 2000)); // 指数バックオフ
        return this.fetchDmmApiData(page, retryCount + 1);
      } else {
        console.error(`❌ API Error (max retries exceeded):`, error.response?.data || error.message);
        throw error;
      }
    }
  }

  async processBatch(items) {
    const results = {
      inserted: 0,
      skipped: 0,
      errors: 0,
      newIds: []
    };

    for (const item of items) {
      try {
        // 高速重複チェック（メモリキャッシュ）
        if (this.existingIds.has(item.content_id)) {
          results.skipped++;
          continue;
        }

        const insertResult = await this.insertVideoToSupabase(item);
        
        if (insertResult.inserted) {
          results.inserted++;
          results.newIds.push(item.content_id);
          this.existingIds.add(item.content_id); // キャッシュに追加
        } else if (insertResult.skipped) {
          results.skipped++;
          this.existingIds.add(item.content_id); // スキップされた場合もキャッシュに追加
        } else if (insertResult.error) {
          results.errors++;
        }

      } catch (error) {
        console.error(`❌ Error processing item ${item.content_id}:`, error.message);
        results.errors++;
      }
    }

    return results;
  }

  async insertVideoToSupabase(item) {
    try {
      // 動画データの変換
      const videoData = {
        external_id: item.content_id,
        title: item.title,
        description: item.description || '',
        duration_seconds: null,
        thumbnail_url: item.imageURL?.large || item.imageURL?.medium || '',
        preview_video_url: item.sampleMovieURL?.size_720_480 || item.sampleMovieURL?.size_560_360 || '',
        distribution_code: item.content_id,
        maker_code: null,
        director: item.iteminfo?.director?.[0]?.name || '',
        series: item.iteminfo?.series?.[0]?.name || '',
        maker: item.iteminfo?.maker?.[0]?.name || '',
        label: item.iteminfo?.label?.[0]?.name || '',
        genre: item.iteminfo?.genre?.[0]?.name || '',
        price: item.prices?.price ? parseInt(String(item.prices.price).replace(/[^0-9]/g, '')) || 0 : 0,
        distribution_started_at: item.date || null,
        product_released_at: item.date || null,
        sample_video_url: item.sampleMovieURL?.size_720_480 || '',
        image_urls: item.sampleImageURL?.sample_s?.image || [],
        source: 'dmm',
        published_at: item.date || new Date().toISOString(),
      };

      // 動画を挿入
      const { data: video, error: videoError } = await supabase
        .from('videos')
        .insert(videoData)
        .select('id')
        .single();

      if (videoError) {
        // 既に存在する場合のエラーハンドリング
        if (videoError.code === '23505') { // Unique constraint violation
          return { skipped: true };
        }
        throw videoError;
      }

      // 関連データの処理（ジャンル・出演者）
      await this.processRelatedData(video.id, item);

      return { inserted: true, video_id: video.id };

    } catch (error) {
      return { error: error.message };
    }
  }

  async processRelatedData(videoId, item) {
    // ジャンル処理
    if (item.iteminfo?.genre && item.iteminfo.genre.length > 0) {
      for (const genreItem of item.iteminfo.genre) {
        if (genreItem.name) {
          await this.insertGenreTag(videoId, genreItem.name);
        }
      }
    }

    // 出演者処理
    if (item.iteminfo?.actress && item.iteminfo.actress.length > 0) {
      for (const actressItem of item.iteminfo.actress) {
        if (actressItem.name) {
          await this.insertPerformer(videoId, actressItem.name);
        }
      }
    }
  }

  async insertGenreTag(videoId, genreName) {
    try {
      // タググループの取得または作成
      let { data: tagGroup } = await supabase
        .from('tag_groups')
        .select('id')
        .eq('name', 'ジャンル')
        .single();

      if (!tagGroup) {
        const { data: newTagGroup } = await supabase
          .from('tag_groups')
          .insert({ name: 'ジャンル' })
          .select('id')
          .single();
        tagGroup = newTagGroup;
      }

      // タグの取得または作成
      let { data: tag } = await supabase
        .from('tags')
        .select('id')
        .eq('name', genreName)
        .single();

      if (!tag) {
        const { data: newTag } = await supabase
          .from('tags')
          .insert({
            name: genreName,
            tag_group_id: tagGroup.id
          })
          .select('id')
          .single();
        tag = newTag;
      }

      // 動画とタグを関連付け
      await supabase
        .from('video_tags')
        .insert({
          video_id: videoId,
          tag_id: tag.id
        });

    } catch (error) {
      // 重複エラーは無視
      if (error.code !== '23505') {
        console.error(`Warning: Failed to insert genre ${genreName}:`, error.message);
      }
    }
  }

  async insertPerformer(videoId, performerName) {
    try {
      // 出演者の取得または作成
      let { data: performer } = await supabase
        .from('performers')
        .select('id')
        .eq('name', performerName)
        .single();

      if (!performer) {
        const { data: newPerformer } = await supabase
          .from('performers')
          .insert({ name: performerName })
          .select('id')
          .single();
        performer = newPerformer;
      }

      // 動画と出演者を関連付け
      await supabase
        .from('video_performers')
        .insert({
          video_id: videoId,
          performer_id: performer.id
        });

    } catch (error) {
      // 重複エラーは無視
      if (error.code !== '23505') {
        console.error(`Warning: Failed to insert performer ${performerName}:`, error.message);
      }
    }
  }

  displayProgress(page) {
    const elapsed = (Date.now() - this.stats.startTime) / 1000;
    const pagesPerSecond = page / elapsed;
    const estimatedTotalTime = CONFIG.estimatedTotalPages / pagesPerSecond;
    const remainingTime = estimatedTotalTime - elapsed;
    
    console.log(`\n📊 Progress Update - Page ${page}/${CONFIG.estimatedTotalPages}`);
    console.log(`   📈 Progress: ${((page / CONFIG.estimatedTotalPages) * 100).toFixed(1)}%`);
    console.log(`   ✅ Inserted: ${this.stats.totalInserted.toLocaleString()}`);
    console.log(`   ⏭️  Skipped: ${this.stats.totalSkipped.toLocaleString()}`);
    console.log(`   ❌ Errors: ${this.stats.totalErrors}`);
    console.log(`   ⏱️  Elapsed: ${(elapsed / 60).toFixed(1)}min`);
    console.log(`   🕐 ETA: ${(remainingTime / 60).toFixed(1)}min`);
    console.log(`   🚀 Speed: ${pagesPerSecond.toFixed(2)} pages/sec`);
  }

  async run() {
    try {
      await this.initialize();
      
      let currentPage = this.progress.currentPage;
      
      while (currentPage <= CONFIG.estimatedTotalPages) {
        try {
          // API呼び出し
          const { items, totalCount } = await this.fetchDmmApiData(currentPage);
          
          if (!items || items.length === 0) {
            console.log('✅ No more items available - sync complete');
            break;
          }

          // バッチ処理
          const batchResults = await this.processBatch(items);
          
          // 統計更新
          this.stats.totalInserted += batchResults.inserted;
          this.stats.totalSkipped += batchResults.skipped;
          this.stats.totalErrors += batchResults.errors;
          this.stats.pagesProcessed = currentPage;

          // 進捗表示
          if (currentPage % 5 === 0) {
            this.displayProgress(currentPage);
          }

          // 進捗保存
          if (currentPage % CONFIG.progressSaveInterval === 0) {
            this.progress.currentPage = currentPage + 1;
            await this.saveProgress();
            console.log(`💾 Progress saved at page ${currentPage}`);
          }

          // IDキャッシュリフレッシュ（定期的）
          if (currentPage % CONFIG.cacheRefreshInterval === 0) {
            await fs.writeFile(CACHE_FILE, JSON.stringify([...this.existingIds], null, 2));
            console.log(`🔄 ID cache updated: ${this.existingIds.size.toLocaleString()} IDs`);
          }

          currentPage++;

          // レート制限
          await new Promise(resolve => setTimeout(resolve, CONFIG.rateLimitDelay));

        } catch (error) {
          console.error(`❌ Error on page ${currentPage}:`, error.message);
          this.stats.totalErrors++;
          
          // エラー時は次のページに進む
          currentPage++;
          
          // エラー時の待機時間を長くする
          await new Promise(resolve => setTimeout(resolve, CONFIG.rateLimitDelay * 2));
        }
      }

      // 最終結果
      await this.displayFinalResults();

    } catch (error) {
      console.error('💥 Critical error:', error.message);
      // 進捗を保存してから終了
      this.progress.currentPage = currentPage;
      await this.saveProgress();
      throw error;
    }
  }

  async displayFinalResults() {
    const duration = (Date.now() - this.stats.startTime) / 1000;
    
    console.log('\n🎉 DMM Bulk Sync Complete!');
    console.log('==========================');
    console.log(`📊 Final Results:`);
    console.log(`   ✅ Total Inserted: ${this.stats.totalInserted.toLocaleString()} videos`);
    console.log(`   ⏭️  Total Skipped: ${this.stats.totalSkipped.toLocaleString()} videos`);
    console.log(`   ❌ Total Errors: ${this.stats.totalErrors}`);
    console.log(`   📄 Pages Processed: ${this.stats.pagesProcessed}`);
    console.log(`   ⏱️  Total Duration: ${(duration / 3600).toFixed(1)} hours`);
    
    // データベース統計確認
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('external_id')
        .eq('source', 'dmm');
      
      if (!error && data) {
        console.log(`\n📈 Final Database State:`);
        console.log(`   🗄️  Total DMM videos: ${data.length.toLocaleString()}`);
      }
    } catch (error) {
      console.log('Could not retrieve final database stats');
    }

    // 完了マーキング
    this.progress.completed = true;
    this.progress.completedAt = new Date().toISOString();
    await this.saveProgress();
  }
}

// 実行
async function main() {
  const sync = new EfficientDmmBulkSync();
  await sync.run();
}

main().catch(console.error);
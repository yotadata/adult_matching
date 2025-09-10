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

// マルチソート戦略設定
const SORT_STRATEGIES = [
  { sort: 'date', maxPages: 200, description: '最新順（最近のデータ重視）' },
  { sort: 'price', maxPages: 150, description: '価格順（古いデータ含む）' },
  { sort: 'review', maxPages: 100, description: 'レビュー順（人気データ）' },
  { sort: 'rank', maxPages: 50, description: 'ランク順（トレンドデータ）' }
];

const CONFIG = {
  batchSize: 100,
  baseDelay: 800,
  maxRetries: 3,
  progressSaveInterval: 10,
  memoryCheckInterval: 25
};

class MultiSortDmmSync {
  constructor() {
    this.existingIds = new Set();
    this.globalStats = {
      totalInserted: 0,
      totalSkipped: 0,
      totalErrors: 0,
      startTime: Date.now()
    };
    this.progressFile = path.join(__dirname, 'multi_sort_progress.json');
    this.cacheFile = path.join(__dirname, 'dmm_ids_cache.json');
  }

  async initialize() {
    console.log('🚀 Multi-Sort DMM Sync - Comprehensive Data Retrieval');
    console.log('=======================================================');
    console.log('📋 Strategy Overview:');
    SORT_STRATEGIES.forEach((strategy, i) => {
      console.log(`   ${i + 1}. ${strategy.sort.toUpperCase()}: ${strategy.maxPages} pages - ${strategy.description}`);
    });
    
    const totalPages = SORT_STRATEGIES.reduce((sum, s) => sum + s.maxPages, 0);
    const estimatedHours = (totalPages * CONFIG.baseDelay) / (1000 * 60 * 60);
    console.log(`\\n🎯 Total Plan: ${totalPages} pages, ~${estimatedHours.toFixed(1)} hours`);
    
    await this.loadExistingIds();
    await this.loadProgress();
  }

  async loadExistingIds() {
    try {
      // キャッシュファイルから読み込み
      const cacheData = await fs.readFile(this.cacheFile, 'utf8');
      const cachedIds = JSON.parse(cacheData);
      this.existingIds = new Set(cachedIds);
      console.log(`\\n✅ Loaded ID cache: ${this.existingIds.size.toLocaleString()} existing IDs`);
    } catch (error) {
      // データベースから新規作成
      console.log('\\n🔄 Creating fresh ID cache from database...');
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
      await fs.writeFile(this.cacheFile, JSON.stringify([...this.existingIds], null, 2));
      
      console.log(`🔄 Created ID cache: ${this.existingIds.size.toLocaleString()} existing IDs`);
    } catch (error) {
      console.error('❌ Error creating ID cache:', error.message);
      throw error;
    }
  }

  async loadProgress() {
    try {
      const progressData = await fs.readFile(this.progressFile, 'utf8');
      this.progress = JSON.parse(progressData);
      console.log(`\\n📁 Loaded existing progress:`);
      this.progress.strategies.forEach((s, i) => {
        const percent = ((s.currentPage / s.maxPages) * 100).toFixed(1);
        const status = s.completed ? '✅' : '🔄';
        console.log(`   ${status} ${s.sort}: ${s.currentPage}/${s.maxPages} (${percent}%)`);
      });
    } catch (error) {
      // 新規プログレス作成
      this.progress = {
        strategies: SORT_STRATEGIES.map(s => ({
          sort: s.sort,
          currentPage: 1,
          maxPages: s.maxPages,
          completed: false,
          inserted: 0,
          skipped: 0,
          errors: 0
        })),
        startedAt: new Date().toISOString(),
        lastSaveAt: null
      };
      console.log('\\n🆕 Created new progress tracking');
    }
  }

  async saveProgress() {
    this.progress.lastSaveAt = new Date().toISOString();
    await fs.writeFile(this.progressFile, JSON.stringify(this.progress, null, 2));
  }

  async fetchDmmApiData(sort, page, retryCount = 0) {
    const params = {
      api_id: DMM_API_CONFIG.api_id,
      affiliate_id: DMM_API_CONFIG.affiliate_id,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      hits: CONFIG.batchSize,
      offset: ((page - 1) * CONFIG.batchSize) + 1,
      sort: sort,
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
        const delay = 1000 * Math.pow(2, retryCount);
        console.log(`⚠️  API Error (retry ${retryCount + 1}): ${error.message} - waiting ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.fetchDmmApiData(sort, page, retryCount + 1);
      } else {
        throw error;
      }
    }
  }

  async processBatch(items, strategyIndex) {
    const results = { inserted: 0, skipped: 0, errors: 0, newIds: [] };
    
    for (const item of items) {
      try {
        // 高速メモリ重複チェック
        if (this.existingIds.has(item.content_id)) {
          results.skipped++;
          continue;
        }

        const insertResult = await this.insertVideoToSupabase(item);
        
        if (insertResult.inserted) {
          results.inserted++;
          results.newIds.push(item.content_id);
          this.existingIds.add(item.content_id);
        } else if (insertResult.skipped) {
          results.skipped++;
          this.existingIds.add(item.content_id);
        } else if (insertResult.error) {
          results.errors++;
        }

      } catch (error) {
        console.error(`❌ Error processing ${item.content_id}:`, error.message);
        results.errors++;
      }
    }

    // 戦略別統計更新
    this.progress.strategies[strategyIndex].inserted += results.inserted;
    this.progress.strategies[strategyIndex].skipped += results.skipped;
    this.progress.strategies[strategyIndex].errors += results.errors;

    return results;
  }

  async insertVideoToSupabase(item) {
    try {
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

      const { data: video, error: videoError } = await supabase
        .from('videos')
        .insert(videoData)
        .select('id')
        .single();

      if (videoError) {
        if (videoError.code === '23505') {
          return { skipped: true };
        }
        throw videoError;
      }

      // 関連データ処理
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

      await supabase
        .from('video_tags')
        .insert({
          video_id: videoId,
          tag_id: tag.id
        });

    } catch (error) {
      if (error.code !== '23505') {
        console.error(`Warning: Failed to insert genre ${genreName}:`, error.message);
      }
    }
  }

  async insertPerformer(videoId, performerName) {
    try {
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

      await supabase
        .from('video_performers')
        .insert({
          video_id: videoId,
          performer_id: performer.id
        });

    } catch (error) {
      if (error.code !== '23505') {
        console.error(`Warning: Failed to insert performer ${performerName}:`, error.message);
      }
    }
  }

  displayStrategyProgress(strategyIndex, page) {
    const strategy = this.progress.strategies[strategyIndex];
    const elapsed = (Date.now() - this.globalStats.startTime) / 1000;
    const progress = (page / strategy.maxPages) * 100;
    
    console.log(`\\n📊 ${strategy.sort.toUpperCase()} Progress - Page ${page}/${strategy.maxPages} (${progress.toFixed(1)}%)`);
    console.log(`   ✅ Inserted: ${strategy.inserted.toLocaleString()}`);
    console.log(`   ⏭️  Skipped: ${strategy.skipped.toLocaleString()}`);
    console.log(`   ❌ Errors: ${strategy.errors}`);
    console.log(`   ⏱️  Elapsed: ${(elapsed / 60).toFixed(1)}min`);
  }

  async runStrategy(strategyIndex) {
    const strategy = this.progress.strategies[strategyIndex];
    
    if (strategy.completed) {
      console.log(`✅ Strategy ${strategy.sort.toUpperCase()} already completed`);
      return;
    }

    console.log(`\\n🎯 Starting Strategy: ${strategy.sort.toUpperCase()} (${strategy.maxPages} pages)`);
    
    for (let page = strategy.currentPage; page <= strategy.maxPages; page++) {
      try {
        const { items, totalCount } = await this.fetchDmmApiData(strategy.sort, page);
        
        if (!items || items.length === 0) {
          console.log(`✅ No more items for ${strategy.sort} - completing strategy`);
          break;
        }

        const batchResults = await this.processBatch(items, strategyIndex);
        
        // グローバル統計更新
        this.globalStats.totalInserted += batchResults.inserted;
        this.globalStats.totalSkipped += batchResults.skipped;
        this.globalStats.totalErrors += batchResults.errors;

        // 進捗表示
        if (page % 5 === 0) {
          this.displayStrategyProgress(strategyIndex, page);
        }

        // 進捗保存
        if (page % CONFIG.progressSaveInterval === 0) {
          strategy.currentPage = page + 1;
          await this.saveProgress();
          
          // ID キャッシュ更新
          if (page % 50 === 0) {
            await fs.writeFile(this.cacheFile, JSON.stringify([...this.existingIds], null, 2));
            console.log(`💾 Cache updated: ${this.existingIds.size.toLocaleString()} IDs`);
          }
        }

        // レート制限
        await new Promise(resolve => setTimeout(resolve, CONFIG.baseDelay));

      } catch (error) {
        console.error(`❌ Error on ${strategy.sort} page ${page}:`, error.message);
        strategy.errors++;
        
        // エラー時は長めに待機
        await new Promise(resolve => setTimeout(resolve, CONFIG.baseDelay * 2));
      }
    }

    // 戦略完了
    strategy.completed = true;
    strategy.currentPage = strategy.maxPages + 1;
    await this.saveProgress();
    
    console.log(`\\n🎉 Strategy ${strategy.sort.toUpperCase()} completed!`);
    console.log(`   ✅ Inserted: ${strategy.inserted.toLocaleString()}`);
    console.log(`   ⏭️  Skipped: ${strategy.skipped.toLocaleString()}`);
    console.log(`   ❌ Errors: ${strategy.errors}`);
  }

  async run() {
    try {
      await this.initialize();
      
      for (let i = 0; i < SORT_STRATEGIES.length; i++) {
        await this.runStrategy(i);
        
        // 戦略間の休憩
        if (i < SORT_STRATEGIES.length - 1) {
          console.log('\\n⏸️  Brief pause between strategies...');
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
      }

      await this.displayFinalResults();

    } catch (error) {
      console.error('💥 Critical error:', error.message);
      await this.saveProgress();
      throw error;
    }
  }

  async displayFinalResults() {
    const duration = (Date.now() - this.globalStats.startTime) / 1000;
    
    console.log('\\n\\n🎉🎉 Multi-Sort DMM Sync Complete! 🎉🎉');
    console.log('============================================');
    console.log(`📊 Global Results:`);
    console.log(`   ✅ Total Inserted: ${this.globalStats.totalInserted.toLocaleString()} videos`);
    console.log(`   ⏭️  Total Skipped: ${this.globalStats.totalSkipped.toLocaleString()} videos`);
    console.log(`   ❌ Total Errors: ${this.globalStats.totalErrors}`);
    console.log(`   ⏱️  Total Duration: ${(duration / 3600).toFixed(1)} hours`);
    
    console.log(`\\n📈 Strategy Breakdown:`);
    this.progress.strategies.forEach(s => {
      const total = s.inserted + s.skipped;
      console.log(`   ${s.sort.toUpperCase()}: ${s.inserted.toLocaleString()} inserted, ${s.skipped.toLocaleString()} skipped (${total.toLocaleString()} total)`);
    });

    // 最終データベース統計
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('external_id')
        .eq('source', 'dmm');
      
      if (!error && data) {
        console.log(`\\n🗄️  Final Database State:`);
        console.log(`   📊 Total DMM videos: ${data.length.toLocaleString()}`);
        console.log(`   📈 Growth: +${(data.length - this.existingIds.size).toLocaleString()} new videos`);
      }
    } catch (error) {
      console.log('Could not retrieve final database stats');
    }

    // 完了マーキング
    this.progress.completedAt = new Date().toISOString();
    this.progress.totalDuration = duration;
    await this.saveProgress();
  }
}

// 実行
async function main() {
  const sync = new MultiSortDmmSync();
  await sync.run();
}

main().catch(console.error);
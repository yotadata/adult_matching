const { createClient } = require('@supabase/supabase-js');
const axios = require('axios');
const fs = require('fs');

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

class MegaDmmSync200k {
  constructor() {
    this.TARGET_VIDEOS = 200000;
    this.existingIds = new Set();
    this.batchCache = new Map();
    this.stats = {
      totalInserted: 0,
      totalSkipped: 0,
      totalErrors: 0,
      apiCallsMade: 0,
      databaseQueries: 0,
      startTime: new Date(),
      lastProgressSave: new Date()
    };
    
    // Mega Sync Strategy - 大規模取得用
    this.MEGA_STRATEGIES = [
      // Phase 1: Core strategies (expanded)
      { sort: 'date', pages: 500, description: '最新順（過去5年分）' },
      { sort: 'price', pages: 800, description: '価格順（全価格帯網羅）' },
      { sort: 'review', pages: 400, description: 'レビュー順（人気作品）' },
      { sort: 'rank', pages: 300, description: 'ランク順（トレンド作品）' },
      
      // Phase 2: Additional coverage strategies  
      { sort: 'date', pages: 1000, offset: 50000, description: '日付順（古い作品）' },
      { sort: 'price', pages: 500, offset: 80000, description: '価格順（レア価格帯）' },
      { sort: 'review', pages: 300, offset: 40000, description: 'レビュー順（隠れた名作）' },
      
      // Phase 3: Deep dive strategies
      { sort: 'rank', pages: 700, offset: 30000, description: 'ランク順（過去ランキング）' }
    ];
    
    this.progressFile = 'scripts/mega_sync_200k_progress.json';
  }

  async initializeSync() {
    console.log('🚀 Mega DMM Sync - 200,000 Videos Target');
    console.log('=========================================\n');
    
    // Load existing progress if available
    await this.loadProgress();
    
    // Create fresh ID cache from database
    console.log('🔄 Creating comprehensive ID cache from database...');
    await this.loadExistingIds();
    console.log(`🔄 Created ID cache: ${this.existingIds.size.toLocaleString()} existing IDs\n`);
    
    // Display strategy overview
    this.displayStrategyOverview();
    
    return true;
  }

  displayStrategyOverview() {
    console.log('📋 Mega Strategy Overview:');
    let totalPages = 0;
    this.MEGA_STRATEGIES.forEach((strategy, i) => {
      totalPages += strategy.pages;
      const offset = strategy.offset ? ` (offset: ${strategy.offset.toLocaleString()})` : '';
      console.log(`   ${i+1}. ${strategy.sort.toUpperCase()}: ${strategy.pages} pages${offset} - ${strategy.description}`);
    });
    
    console.log(`\n🎯 Total Plan: ${totalPages.toLocaleString()} pages`);
    console.log(`📊 Expected Duration: ~${Math.ceil(totalPages * 1.2 / 60)} hours`);
    console.log(`🎯 Target: ${this.TARGET_VIDEOS.toLocaleString()} videos\n`);
  }

  async loadExistingIds() {
    // Get accurate count first
    const { data: countResult, error: countError } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm');

    if (countError) throw countError;

    const actualCount = countResult[0].count;
    console.log(`📊 Verified database count: ${actualCount.toLocaleString()} videos`);

    // Load external IDs for duplicate checking
    const { data: videos, error } = await supabase
      .from('videos')
      .select('external_id')
      .eq('source', 'dmm');

    if (error) throw error;

    this.existingIds.clear();
    videos.forEach(video => {
      this.existingIds.add(video.external_id);
    });

    console.log(`📊 Current database state: ${this.existingIds.size.toLocaleString()} unique IDs cached`);
    console.log(`🎯 Remaining target: ${(this.TARGET_VIDEOS - actualCount).toLocaleString()} videos needed\n`);
    
    // Update stats with actual count
    this.stats.currentDbCount = actualCount;
  }

  async loadProgress() {
    try {
      if (fs.existsSync(this.progressFile)) {
        const progress = JSON.parse(fs.readFileSync(this.progressFile, 'utf8'));
        console.log(`📂 Resuming from saved progress: ${progress.completedStrategies} strategies completed`);
        this.stats = { ...this.stats, ...progress.stats };
        return progress;
      }
    } catch (error) {
      console.log('⚠️  No previous progress found, starting fresh');
    }
    return null;
  }

  async saveProgress(currentStrategy, completedStrategies) {
    const progress = {
      currentStrategy,
      completedStrategies,
      stats: this.stats,
      lastSaveAt: new Date().toISOString(),
      totalProgress: `${Math.round((completedStrategies / this.MEGA_STRATEGIES.length) * 100)}%`
    };
    
    fs.writeFileSync(this.progressFile, JSON.stringify(progress, null, 2));
    this.stats.lastProgressSave = new Date();
  }

  async fetchDmmApiData(sort, page, limit = 100, offset = 0) {
    const actualOffset = offset + ((page - 1) * limit) + 1;
    
    const params = {
      api_id: DMM_API_CONFIG.api_id,
      affiliate_id: DMM_API_CONFIG.affiliate_id,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      hits: limit,
      offset: actualOffset,
      sort: sort,
      output: 'json'
    };

    this.stats.apiCallsMade++;
    
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
      this.stats.totalErrors++;
      throw error;
    }
  }

  isDuplicate(contentId) {
    return this.existingIds.has(contentId) || this.batchCache.has(contentId);
  }

  async insertVideoToDatabase(item) {
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
        .select('id, external_id')
        .single();

      if (videoError) {
        if (videoError.code === '23505') { // Unique constraint violation
          this.stats.totalSkipped++;
          return { skipped: true, reason: 'database_duplicate' };
        }
        throw videoError;
      }

      // Add to caches
      this.existingIds.add(item.content_id);
      this.batchCache.set(item.content_id, video.id);
      this.stats.totalInserted++;
      
      return { inserted: true, video_id: video.id };

    } catch (error) {
      this.stats.totalErrors++;
      return { error: error.message };
    }
  }

  async executeStrategy(strategy, strategyIndex) {
    console.log(`\n🎯 Starting Strategy ${strategyIndex + 1}/${this.MEGA_STRATEGIES.length}: ${strategy.sort.toUpperCase()}`);
    console.log(`📋 ${strategy.description} - ${strategy.pages} pages`);
    
    const startInserted = this.stats.totalInserted;
    const startSkipped = this.stats.totalSkipped;
    
    for (let page = 1; page <= strategy.pages; page++) {
      try {
        // Fetch data from API
        const { items } = await this.fetchDmmApiData(strategy.sort, page, 100, strategy.offset || 0);
        
        // Process each item
        for (const item of items) {
          if (this.isDuplicate(item.content_id)) {
            this.stats.totalSkipped++;
            continue;
          }
          
          const result = await this.insertVideoToDatabase(item);
          
          // Rate limiting
          await new Promise(resolve => setTimeout(resolve, 800));
        }
        
        // Progress reporting
        if (page % 25 === 0) {
          const elapsed = (new Date() - this.stats.startTime) / (1000 * 60);
          const currentTotal = this.stats.currentDbCount + this.stats.totalInserted;
          const remaining = this.TARGET_VIDEOS - currentTotal;
          const progress = ((currentTotal / this.TARGET_VIDEOS) * 100).toFixed(1);
          
          console.log(`📊 ${strategy.sort.toUpperCase()} Progress - Page ${page}/${strategy.pages} (${((page/strategy.pages)*100).toFixed(1)}%)`);
          console.log(`   📊 Current Total: ${currentTotal.toLocaleString()} videos (${progress}% of target)`);
          console.log(`   🎯 Remaining: ${remaining.toLocaleString()} videos needed`);
          console.log(`   ✅ This strategy: +${this.stats.totalInserted - startInserted} inserted, +${this.stats.totalSkipped - startSkipped} skipped`);
          console.log(`   ⏱️  Total elapsed: ${elapsed.toFixed(1)}min, API calls: ${this.stats.apiCallsMade.toLocaleString()}\n`);
          
          // Save progress
          if (page % 50 === 0) {
            await this.saveProgress(strategyIndex, strategyIndex);
            console.log('💾 Progress saved\n');
          }
        }
        
        // Early completion check
        if (this.existingIds.size >= this.TARGET_VIDEOS) {
          console.log('🎉 TARGET REACHED! Stopping early.');
          break;
        }
        
      } catch (error) {
        console.error(`❌ Error on page ${page}:`, error.message);
        // Continue with exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.min(10000, 1000 * Math.pow(2, this.stats.totalErrors))));
      }
    }
    
    const strategyInserted = this.stats.totalInserted - startInserted;
    const strategySkipped = this.stats.totalSkipped - startSkipped;
    
    console.log(`\n🎉 Strategy ${strategy.sort.toUpperCase()} completed!`);
    console.log(`   ✅ Inserted: ${strategyInserted.toLocaleString()}`);
    console.log(`   ⏭️  Skipped: ${strategySkipped.toLocaleString()}`);
    
    return { inserted: strategyInserted, skipped: strategySkipped };
  }

  async runMegaSync() {
    try {
      await this.initializeSync();
      
      let completedStrategies = 0;
      
      for (let i = 0; i < this.MEGA_STRATEGIES.length; i++) {
        if (this.existingIds.size >= this.TARGET_VIDEOS) {
          console.log('\n🎉 TARGET OF 200,000 VIDEOS REACHED!');
          break;
        }
        
        await this.executeStrategy(this.MEGA_STRATEGIES[i], i);
        completedStrategies++;
        
        await this.saveProgress(i + 1, completedStrategies);
        
        // Brief pause between strategies
        console.log('⏸️  Brief pause between strategies...\n');
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
      
      // Final results
      await this.displayFinalResults();
      
    } catch (error) {
      console.error('💥 Mega sync failed:', error.message);
      throw error;
    }
  }

  async displayFinalResults() {
    const totalTime = (new Date() - this.stats.startTime) / (1000 * 60 * 60);
    
    // Get final database count
    const { data: finalCount } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm');

    console.log('\n\n🎉🎉🎉 MEGA DMM SYNC COMPLETE! 🎉🎉🎉');
    console.log('=============================================');
    console.log(`📊 Final Results:`);
    console.log(`   🎯 Target: ${this.TARGET_VIDEOS.toLocaleString()} videos`);
    console.log(`   ✅ Achieved: ${finalCount[0].count.toLocaleString()} videos`);
    console.log(`   📈 Success Rate: ${((finalCount[0].count / this.TARGET_VIDEOS) * 100).toFixed(1)}%`);
    console.log(`   ⏱️  Total Duration: ${totalTime.toFixed(1)} hours`);
    console.log(`   🌐 API Calls Made: ${this.stats.apiCallsMade.toLocaleString()}`);
    console.log(`   ✅ Total Inserted: ${this.stats.totalInserted.toLocaleString()}`);
    console.log(`   ⏭️  Total Skipped: ${this.stats.totalSkipped.toLocaleString()}`);
    console.log(`   ❌ Total Errors: ${this.stats.totalErrors}`);
    
    if (finalCount[0].count >= this.TARGET_VIDEOS) {
      console.log('\n✅ MISSION ACCOMPLISHED: 200,000+ videos successfully retrieved!');
    } else {
      console.log(`\n⚠️  Target not reached. ${(this.TARGET_VIDEOS - finalCount[0].count).toLocaleString()} videos remaining.`);
    }
  }
}

// 実行
async function main() {
  const megaSync = new MegaDmmSync200k();
  await megaSync.runMegaSync();
}

main().catch(console.error);
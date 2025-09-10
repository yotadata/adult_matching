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

class Efficient200kSync {
  constructor() {
    this.TARGET_VIDEOS = 200000;
    this.CURRENT_COUNT = 0;
    this.stats = {
      totalInserted: 0,
      totalSkipped: 0,
      totalErrors: 0,
      apiCallsMade: 0,
      startTime: new Date()
    };
    
    // Efficient strategy for 200k - focus on maximum coverage with minimal duplication
    this.STRATEGIES = [
      // Core expansion strategies
      { sort: 'date', pages: 600, description: '最新順・拡張' },
      { sort: 'price', pages: 1000, description: '価格順・全価格帯' },
      { sort: 'review', pages: 500, description: 'レビュー順・人気作品' },
      { sort: 'rank', pages: 400, description: 'ランク順・トレンド' },
      
      // Deep historical data
      { sort: 'date', pages: 800, offset: 60000, description: '日付順・過去データ' },
      { sort: 'price', pages: 600, offset: 100000, description: '価格順・レア価格帯' },
      { sort: 'review', pages: 400, offset: 50000, description: 'レビュー順・隠れた名作' },
      { sort: 'rank', pages: 300, offset: 40000, description: 'ランク順・過去ランキング' }
    ];
  }

  async getCurrentDbCount() {
    const { data: countResult, error } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm');
    
    if (error) throw error;
    return countResult[0].count;
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

      return {
        items: response.data?.result?.items || [],
        totalCount: response.data?.result?.total_count || 0
      };
    } catch (error) {
      this.stats.totalErrors++;
      throw error;
    }
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
        .select('id')
        .single();

      if (videoError) {
        if (videoError.code === '23505') { // Unique constraint violation
          this.stats.totalSkipped++;
          return { skipped: true };
        }
        throw videoError;
      }

      this.stats.totalInserted++;
      return { inserted: true, video_id: video.id };

    } catch (error) {
      this.stats.totalErrors++;
      return { error: error.message };
    }
  }

  async executeStrategy(strategy, index) {
    console.log(`\n🎯 Strategy ${index + 1}/${this.STRATEGIES.length}: ${strategy.sort.toUpperCase()}`);
    console.log(`📋 ${strategy.description} - ${strategy.pages} pages`);
    
    const startInserted = this.stats.totalInserted;
    const startSkipped = this.stats.totalSkipped;
    
    for (let page = 1; page <= strategy.pages; page++) {
      try {
        const { items } = await this.fetchDmmApiData(strategy.sort, page, 100, strategy.offset || 0);
        
        for (const item of items) {
          const result = await this.insertVideoToDatabase(item);
          
          // Rate limiting
          await new Promise(resolve => setTimeout(resolve, 600));
        }
        
        // Progress check every 50 pages
        if (page % 50 === 0) {
          const currentDbCount = await this.getCurrentDbCount();
          const elapsed = (new Date() - this.stats.startTime) / (1000 * 60);
          const remaining = this.TARGET_VIDEOS - currentDbCount;
          const progress = ((currentDbCount / this.TARGET_VIDEOS) * 100).toFixed(1);
          
          console.log(`📊 ${strategy.sort.toUpperCase()} - Page ${page}/${strategy.pages} (${((page/strategy.pages)*100).toFixed(1)}%)`);
          console.log(`   📊 Total DB Count: ${currentDbCount.toLocaleString()} videos (${progress}% of 200k target)`);
          console.log(`   🎯 Remaining: ${remaining.toLocaleString()} videos needed`);
          console.log(`   ✅ Strategy progress: +${this.stats.totalInserted - startInserted} inserted, +${this.stats.totalSkipped - startSkipped} skipped`);
          console.log(`   ⏱️  Elapsed: ${elapsed.toFixed(1)}min, API: ${this.stats.apiCallsMade.toLocaleString()} calls\n`);
          
          // Check if target reached
          if (currentDbCount >= this.TARGET_VIDEOS) {
            console.log('🎉 TARGET OF 200,000 VIDEOS REACHED!');
            return { targetReached: true };
          }
        }
        
      } catch (error) {
        console.error(`❌ Error on page ${page}:`, error.message);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
    
    console.log(`\n✅ Strategy ${strategy.sort.toUpperCase()} completed!`);
    console.log(`   Inserted: ${this.stats.totalInserted - startInserted}, Skipped: ${this.stats.totalSkipped - startSkipped}`);
    
    return { completed: true };
  }

  async run() {
    console.log('🚀 Efficient 200k DMM Sync');
    console.log('==========================\n');
    
    // Check starting point
    const startingCount = await this.getCurrentDbCount();
    console.log(`📊 Starting database count: ${startingCount.toLocaleString()} videos`);
    console.log(`🎯 Target: ${this.TARGET_VIDEOS.toLocaleString()} videos`);
    console.log(`📈 Remaining needed: ${(this.TARGET_VIDEOS - startingCount).toLocaleString()} videos\n`);
    
    if (startingCount >= this.TARGET_VIDEOS) {
      console.log('✅ Target already achieved!');
      return;
    }
    
    // Execute strategies
    for (let i = 0; i < this.STRATEGIES.length; i++) {
      const result = await this.executeStrategy(this.STRATEGIES[i], i);
      
      if (result.targetReached) {
        console.log('\n🎉 MISSION ACCOMPLISHED: 200,000 videos reached!');
        break;
      }
      
      // Brief pause between strategies
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
    
    // Final results
    const finalCount = await this.getCurrentDbCount();
    const totalTime = (new Date() - this.stats.startTime) / (1000 * 60 * 60);
    
    console.log('\n\n🎉 EFFICIENT 200K SYNC COMPLETE!');
    console.log('=================================');
    console.log(`📊 Final count: ${finalCount.toLocaleString()} videos`);
    console.log(`🎯 Target achievement: ${((finalCount / this.TARGET_VIDEOS) * 100).toFixed(1)}%`);
    console.log(`⏱️  Total time: ${totalTime.toFixed(1)} hours`);
    console.log(`🌐 API calls: ${this.stats.apiCallsMade.toLocaleString()}`);
    console.log(`✅ Inserted: ${this.stats.totalInserted.toLocaleString()}`);
    console.log(`⏭️  Skipped: ${this.stats.totalSkipped.toLocaleString()}`);
    console.log(`❌ Errors: ${this.stats.totalErrors}`);
    
    if (finalCount >= this.TARGET_VIDEOS) {
      console.log('\n✅ SUCCESS: 200,000+ video target achieved!');
    }
  }
}

async function main() {
  const sync = new Efficient200kSync();
  await sync.run();
}

main().catch(console.error);
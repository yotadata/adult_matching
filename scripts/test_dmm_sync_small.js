const { createClient } = require('@supabase/supabase-js');
const axios = require('axios');

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

class SmallTestSync {
  constructor() {
    this.stats = {
      totalInserted: 0,
      totalSkipped: 0,
      totalErrors: 0,
      apiCallsMade: 0,
      databaseQueries: 0
    };
  }

  async fetchDmmApiData(sort, page, limit = 10) {
    const params = {
      api_id: DMM_API_CONFIG.api_id,
      affiliate_id: DMM_API_CONFIG.affiliate_id,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      hits: limit,
      offset: ((page - 1) * limit) + 1,
      sort: sort,
      output: 'json'
    };

    this.stats.apiCallsMade++;
    console.log(`🌐 API Call #${this.stats.apiCallsMade}: ${sort} page ${page}, limit ${limit}`);

    try {
      const response = await axios.get(DMM_API_CONFIG.baseUrl, { 
        params: params,
        timeout: 30000
      });

      if (response.data?.result?.items) {
        console.log(`✅ API Response: ${response.data.result.items.length} items received`);
        return {
          items: response.data.result.items,
          totalCount: response.data.result.total_count || 0
        };
      } else {
        throw new Error('Invalid API response format');
      }
    } catch (error) {
      console.error(`❌ API Error:`, error.response?.data || error.message);
      throw error;
    }
  }

  async checkDuplicateInDatabase(contentId) {
    this.stats.databaseQueries++;
    console.log(`🔍 DB Query #${this.stats.databaseQueries}: Checking duplicate for ${contentId}`);

    const { data: existing, error } = await supabase
      .from('videos')
      .select('id')
      .eq('external_id', contentId)
      .eq('source', 'dmm')
      .single();

    if (error && error.code !== 'PGRST116') { // PGRST116 is "not found" error
      throw error;
    }

    const isDuplicate = !!existing;
    console.log(`   ${isDuplicate ? '❌' : '✅'} ${contentId}: ${isDuplicate ? 'EXISTS' : 'NEW'}`);
    return isDuplicate;
  }

  async insertVideoToDatabase(item) {
    console.log(`💾 Inserting: ${item.content_id} - ${item.title?.substring(0, 40)}...`);

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
        .select('id, external_id, title')
        .single();

      if (videoError) {
        if (videoError.code === '23505') { // Unique constraint violation
          console.log(`   ⚠️  Database constraint: ${item.content_id} already exists`);
          return { skipped: true, reason: 'database_duplicate' };
        }
        throw videoError;
      }

      console.log(`   ✅ Successfully inserted: ${video.external_id} (DB ID: ${video.id})`);
      return { inserted: true, video_id: video.id, video_data: video };

    } catch (error) {
      console.error(`   ❌ Database error for ${item.content_id}:`, error.message);
      return { error: error.message };
    }
  }

  async runSmallTest() {
    console.log('🧪 Small DMM API Sync Test');
    console.log('==========================\n');

    try {
      // Test 1: フェッチテスト (10件のみ)
      console.log('📋 Test 1: Fetching 10 latest videos...');
      const { items, totalCount } = await this.fetchDmmApiData('date', 1, 10);
      
      console.log(`📊 API Results:`);
      console.log(`   Total available: ${totalCount.toLocaleString()}`);
      console.log(`   Fetched: ${items.length}`);
      console.log('');

      // Test 2: 重複チェックテスト
      console.log('📋 Test 2: Checking duplicates for each item...');
      const duplicateResults = [];
      for (const item of items) {
        const isDuplicate = await this.checkDuplicateInDatabase(item.content_id);
        duplicateResults.push({ contentId: item.content_id, isDuplicate });
      }
      console.log('');

      // Test 3: 新規データのみ挿入テスト
      console.log('📋 Test 3: Inserting new videos only...');
      const newItems = items.filter((_, i) => !duplicateResults[i].isDuplicate);
      
      if (newItems.length === 0) {
        console.log('⚠️  No new items to insert - all are duplicates');
      } else {
        console.log(`📝 Attempting to insert ${newItems.length} new videos:`);
        
        for (const item of newItems) {
          const result = await this.insertVideoToDatabase(item);
          
          if (result.inserted) {
            this.stats.totalInserted++;
          } else if (result.skipped) {
            this.stats.totalSkipped++;
            console.log(`   ℹ️  Skipped reason: ${result.reason}`);
          } else if (result.error) {
            this.stats.totalErrors++;
          }
          
          // Wait between inserts
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

      // Test 4: データベース確認
      console.log('\n📋 Test 4: Verifying database state...');
      const { data: allVideos, error } = await supabase
        .from('videos')
        .select('external_id, created_at')
        .eq('source', 'dmm')
        .order('created_at', { ascending: false });

      if (error) {
        throw error;
      }

      const today = new Date().toISOString().split('T')[0];
      const todayVideos = allVideos.filter(v => v.created_at.startsWith(today));

      console.log('🗄️  Database State:');
      console.log(`   Total DMM videos: ${allVideos.length}`);
      console.log(`   Added today: ${todayVideos.length}`);

      // Final Results
      console.log('\n🎯 Test Results Summary:');
      console.log('========================');
      console.log(`📊 Statistics:`);
      console.log(`   API Calls Made: ${this.stats.apiCallsMade}`);
      console.log(`   Database Queries: ${this.stats.databaseQueries}`);
      console.log(`   Videos Inserted: ${this.stats.totalInserted}`);
      console.log(`   Videos Skipped: ${this.stats.totalSkipped}`);
      console.log(`   Errors: ${this.stats.totalErrors}`);

      console.log(`\n📋 Duplicate Analysis:`);
      const existingCount = duplicateResults.filter(r => r.isDuplicate).length;
      const newCount = duplicateResults.filter(r => !r.isDuplicate).length;
      console.log(`   Existing in DB: ${existingCount}/${items.length}`);
      console.log(`   New items: ${newCount}/${items.length}`);

      if (this.stats.totalInserted > 0) {
        console.log('\n✅ Test PASSED: Successfully inserted new videos');
        console.log(`🔍 Sample inserted IDs: ${allVideos.slice(0, Math.min(3, this.stats.totalInserted)).map(v => v.external_id).join(', ')}`);
      } else if (newCount === 0) {
        console.log('\n⚠️  Test INCONCLUSIVE: No new videos available to test insertion');
      } else {
        console.log('\n❌ Test FAILED: No videos were inserted despite having new items');
      }

    } catch (error) {
      console.error('\n💥 Test FAILED with error:', error.message);
      throw error;
    }
  }
}

// 実行
async function main() {
  const testSync = new SmallTestSync();
  await testSync.runSmallTest();
}

main().catch(console.error);
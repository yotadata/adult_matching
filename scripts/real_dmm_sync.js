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

async function fetchDmmApiData(page = 1, limit = 50) {
  const params = {
    api_id: DMM_API_CONFIG.api_id,
    affiliate_id: DMM_API_CONFIG.affiliate_id,
    site: 'FANZA',
    service: 'digital',
    floor: 'videoa',
    hits: limit,
    offset: ((page - 1) * limit) + 1,
    sort: 'date',
    output: 'json'
  };

  console.log(`🔍 Fetching DMM API data: page ${page}, limit ${limit}`);
  
  try {
    const response = await axios.get(DMM_API_CONFIG.baseUrl, { 
      params: params,
      timeout: 30000
    });

    if (response.data?.result?.items) {
      console.log(`✅ Retrieved ${response.data.result.items.length} items from DMM API`);
      return {
        items: response.data.result.items,
        totalCount: response.data.result.total_count || 0
      };
    } else {
      throw new Error('Invalid API response format');
    }
  } catch (error) {
    console.error(`❌ DMM API Error:`, error.response?.data || error.message);
    throw error;
  }
}

async function insertVideoToSupabase(item) {
  try {
    // 動画データの変換
    const videoData = {
      external_id: item.content_id,
      title: item.title,
      description: item.description || '',
      duration_seconds: null, // DMM APIには通常duration情報がない
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

    // 重複チェック
    const { data: existing } = await supabase
      .from('videos')
      .select('id')
      .eq('external_id', videoData.external_id)
      .eq('source', 'dmm')
      .single();

    if (existing) {
      console.log(`⏭️  Skipping duplicate: ${videoData.external_id}`);
      return { skipped: true };
    }

    // 動画を挿入
    const { data: video, error: videoError } = await supabase
      .from('videos')
      .insert(videoData)
      .select('id')
      .single();

    if (videoError) {
      throw videoError;
    }

    // ジャンル・タグの処理
    if (item.iteminfo?.genre && item.iteminfo.genre.length > 0) {
      for (const genreItem of item.iteminfo.genre) {
        if (genreItem.name) {
          await insertGenreTag(video.id, genreItem.name);
        }
      }
    }

    // 出演者の処理
    if (item.iteminfo?.actress && item.iteminfo.actress.length > 0) {
      for (const actressItem of item.iteminfo.actress) {
        if (actressItem.name) {
          await insertPerformer(video.id, actressItem.name);
        }
      }
    }

    console.log(`✅ Inserted: ${videoData.title} (${videoData.external_id})`);
    return { inserted: true, video_id: video.id };

  } catch (error) {
    console.error(`❌ Error inserting video ${item.content_id}:`, error.message);
    return { error: error.message };
  }
}

async function insertGenreTag(videoId, genreName) {
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
    console.error(`Warning: Failed to insert genre ${genreName}:`, error.message);
  }
}

async function insertPerformer(videoId, performerName) {
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
    console.error(`Warning: Failed to insert performer ${performerName}:`, error.message);
  }
}

async function main() {
  console.log('🚀 Starting Real DMM API Data Sync');
  console.log('================================');
  
  const startTime = Date.now();
  let totalInserted = 0;
  let totalSkipped = 0;
  let totalErrors = 0;
  let page = 1;
  const limit = 100; // 1回で100件取得
  const maxPages = 50; // 最大50ページ（5000件）に拡大
  
  try {
    for (page = 1; page <= maxPages; page++) {
      console.log(`\\n📄 Processing page ${page}/${maxPages}...`);
      
      const { items, totalCount } = await fetchDmmApiData(page, limit);
      
      if (!items || items.length === 0) {
        console.log('No more items available');
        break;
      }
      
      console.log(`Processing ${items.length} videos...`);
      
      for (const item of items) {
        const result = await insertVideoToSupabase(item);
        
        if (result.inserted) totalInserted++;
        else if (result.skipped) totalSkipped++;
        else if (result.error) totalErrors++;
      }
      
      console.log(`Page ${page} summary: +${items.length} processed`);
      
      // API rate limiting対策とプログレス表示
      if (page < maxPages) {
        const progress = ((page / maxPages) * 100).toFixed(1);
        console.log(`⏱️  Progress: ${progress}% - Waiting 1 second before next page...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
  } catch (error) {
    console.error('💥 Sync failed:', error.message);
  }
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  
  console.log('\\n🎉 DMM API Data Sync Complete!');
  console.log('================================');
  console.log(`📊 Results:`);
  console.log(`   ✅ Inserted: ${totalInserted} videos`);
  console.log(`   ⏭️  Skipped: ${totalSkipped} videos`);
  console.log(`   ❌ Errors: ${totalErrors} videos`);
  console.log(`   ⏱️  Duration: ${duration}s`);
  console.log(`   📄 Pages processed: ${page - 1}`);
  
  // データベース統計確認
  try {
    const { count } = await supabase
      .from('videos')
      .select('count')
      .eq('source', 'dmm')
      .single();
    
    console.log(`\\n📈 Total DMM videos in database: ${count}`);
  } catch (error) {
    console.log('Could not retrieve database stats');
  }
}

// 実行
main().catch(console.error);
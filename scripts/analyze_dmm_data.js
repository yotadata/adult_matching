const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = 'http://127.0.0.1:54321';
const serviceRoleKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, serviceRoleKey);

async function analyzeDataQuality() {
  console.log('🔍 DMM Data Quality Analysis');
  console.log('============================\n');
  
  try {
    // 1. 基本統計
    const { data: allVideos, error } = await supabase
      .from('videos')
      .select('*')
      .eq('source', 'dmm');
    
    if (error) throw error;
    
    console.log('📊 Basic Statistics:');
    console.log(`  Total DMM videos: ${allVideos.length}`);
    
    // 2. データソース別統計
    const testVideos = allVideos.filter(v => v.external_id.includes('test') || v.title.includes('テスト'));
    const realVideos = allVideos.filter(v => !v.external_id.includes('test') && !v.title.includes('テスト'));
    
    console.log(`  Test videos: ${testVideos.length}`);
    console.log(`  Real API videos: ${realVideos.length}`);
    
    // 3. データ完全性チェック
    const missingData = {
      title: allVideos.filter(v => !v.title || v.title.trim() === '').length,
      external_id: allVideos.filter(v => !v.external_id).length,
      thumbnail_url: allVideos.filter(v => !v.thumbnail_url).length,
      price: allVideos.filter(v => v.price === null || v.price === undefined).length,
      genre: allVideos.filter(v => !v.genre || v.genre.trim() === '').length,
      maker: allVideos.filter(v => !v.maker || v.maker.trim() === '').length,
      director: allVideos.filter(v => !v.director || v.director.trim() === '').length,
      series: allVideos.filter(v => !v.series || v.series.trim() === '').length
    };
    
    console.log('\n❌ Missing Data Analysis:');
    Object.entries(missingData).forEach(([field, count]) => {
      const percentage = ((count / allVideos.length) * 100).toFixed(1);
      console.log(`  ${field}: ${count} missing (${percentage}%)`);
    });
    
    // 4. 価格分析
    const prices = allVideos.map(v => v.price).filter(p => p !== null && p !== undefined);
    const avgPrice = prices.length > 0 ? (prices.reduce((a, b) => a + b, 0) / prices.length).toFixed(0) : 0;
    const minPrice = prices.length > 0 ? Math.min(...prices) : 0;
    const maxPrice = prices.length > 0 ? Math.max(...prices) : 0;
    
    console.log('\n💰 Price Analysis:');
    console.log(`  Average price: ¥${avgPrice}`);
    console.log(`  Price range: ¥${minPrice} - ¥${maxPrice}`);
    console.log(`  Videos with price: ${prices.length}/${allVideos.length}`);
    
    // 5. ジャンル分析
    const genreCount = {};
    allVideos.forEach(v => {
      if (v.genre && v.genre.trim() !== '') {
        genreCount[v.genre] = (genreCount[v.genre] || 0) + 1;
      }
    });
    
    console.log('\n🎭 Top Genres:');
    const sortedGenres = Object.entries(genreCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10);
    sortedGenres.forEach(([genre, count]) => {
      console.log(`  ${genre}: ${count} videos`);
    });
    
    // 6. メーカー分析
    const makerCount = {};
    allVideos.forEach(v => {
      if (v.maker && v.maker.trim() !== '') {
        makerCount[v.maker] = (makerCount[v.maker] || 0) + 1;
      }
    });
    
    console.log('\n🏢 Top Makers:');
    const sortedMakers = Object.entries(makerCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10);
    sortedMakers.forEach(([maker, count]) => {
      console.log(`  ${maker}: ${count} videos`);
    });
    
    // 7. 最新動画サンプル
    console.log('\n📋 Recent Real API Videos (Sample):');
    const recentRealVideos = realVideos
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      .slice(0, 5);
    
    recentRealVideos.forEach((video, i) => {
      console.log(`\n  --- Sample ${i+1} ---`);
      console.log(`  ID: ${video.external_id}`);
      console.log(`  Title: ${video.title.substring(0, 80)}...`);
      console.log(`  Genre: ${video.genre || 'N/A'}`);
      console.log(`  Maker: ${video.maker || 'N/A'}`);
      console.log(`  Director: ${video.director || 'N/A'}`);
      console.log(`  Price: ¥${video.price || 'N/A'}`);
      console.log(`  Series: ${video.series || 'N/A'}`);
      console.log(`  Image URLs: ${Array.isArray(video.image_urls) ? video.image_urls.length : 0} items`);
      console.log(`  Created: ${video.created_at}`);
    });
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

// 関連テーブル分析
async function analyzeRelatedTables() {
  console.log('\n\n🔗 Related Tables Analysis');
  console.log('============================\n');
  
  try {
    // Tags analysis
    const { data: tags, error: tagsError } = await supabase
      .from('tags')
      .select('*');
    
    if (!tagsError) {
      console.log(`📊 Tags: ${tags.length} total`);
      const tagSample = tags.slice(0, 10);
      console.log('  Sample tags:', tagSample.map(t => t.name).join(', '));
    }
    
    // Performers analysis
    const { data: performers, error: performersError } = await supabase
      .from('performers')
      .select('*');
    
    if (!performersError) {
      console.log(`\n👩‍🎭 Performers: ${performers.length} total`);
      const performerSample = performers.slice(0, 10);
      console.log('  Sample performers:', performerSample.map(p => p.name).join(', '));
    }
    
    // Video-Tags relationships
    const { data: videoTags, error: videoTagsError } = await supabase
      .from('video_tags')
      .select('*');
    
    if (!videoTagsError) {
      console.log(`\n🏷️  Video-Tags relationships: ${videoTags.length} total`);
    }
    
    // Video-Performers relationships
    const { data: videoPerformers, error: videoPerformersError } = await supabase
      .from('video_performers')
      .select('*');
    
    if (!videoPerformersError) {
      console.log(`👥 Video-Performers relationships: ${videoPerformers.length} total`);
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

async function main() {
  await analyzeDataQuality();
  await analyzeRelatedTables();
  
  console.log('\n✅ Analysis completed!');
}

main().catch(console.error);
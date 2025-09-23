const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');

// Supabase設定
const supabaseUrl = 'http://127.0.0.1:54321';
const serviceRoleKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, serviceRoleKey);

class AccurateContentLinker {
  constructor() {
    this.reviewData = null;
    this.apiVideoCount = 0;
    this.matchedIds = new Set();
    this.stats = {
      totalReviews: 0,
      uniqueContentIds: 0,
      totalApiVideos: 0,
      matchedVideos: 0,
      linkingRate: 0,
      totalUserActions: 0
    };
  }

  async loadReviewData() {
    console.log('📋 Loading review data...');
    const reviewFile = 'data_processing/processed_data/integrated_reviews.json';
    const rawData = fs.readFileSync(reviewFile, 'utf8');
    this.reviewData = JSON.parse(rawData);
    
    this.stats.totalReviews = this.reviewData.length;
    const uniqueIds = new Set(this.reviewData.map(r => r.content_id));
    this.stats.uniqueContentIds = uniqueIds.size;
    
    console.log(`✅ Loaded ${this.stats.totalReviews.toLocaleString()} reviews`);
    console.log(`📊 Unique Content IDs: ${this.stats.uniqueContentIds.toLocaleString()}\n`);
  }

  async getAccurateApiVideoCount() {
    console.log('🎥 Getting accurate API video count...');
    
    const { data: countResult, error } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm');
    
    if (error) throw error;
    
    this.stats.totalApiVideos = countResult[0].count;
    console.log(`✅ Total API videos: ${this.stats.totalApiVideos.toLocaleString()}\n`);
  }

  async performDirectLinking() {
    console.log('🔗 Performing direct Content ID matching via database queries...');
    
    // レビューデータのユニークなcontent_idリストを作成
    const uniqueContentIds = [...new Set(this.reviewData.map(r => r.content_id))];
    console.log(`📊 Checking ${uniqueContentIds.length.toLocaleString()} unique content IDs...`);
    
    // バッチでデータベースクエリを実行
    const batchSize = 100;
    let totalMatched = 0;
    const matchedVideos = new Set();
    
    for (let i = 0; i < uniqueContentIds.length; i += batchSize) {
      const batch = uniqueContentIds.slice(i, i + batchSize);
      
      const { data: matches, error } = await supabase
        .from('videos')
        .select('external_id')
        .eq('source', 'dmm')
        .in('external_id', batch);
      
      if (error) throw error;
      
      matches.forEach(video => {
        matchedVideos.add(video.external_id);
      });
      
      console.log(`   Batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(uniqueContentIds.length/batchSize)}: ${matches.length} matches found`);
    }
    
    this.matchedIds = matchedVideos;
    this.stats.matchedVideos = matchedVideos.size;
    this.stats.linkingRate = (matchedVideos.size / this.stats.uniqueContentIds * 100);
    
    console.log(`\n📊 Direct Linking Results:`);
    console.log(`   ✅ Matched videos: ${matchedVideos.size.toLocaleString()} (${this.stats.linkingRate.toFixed(1)}%)`);
    console.log(`   ❌ Unmatched: ${(this.stats.uniqueContentIds - matchedVideos.size).toLocaleString()} content IDs`);
  }

  async generateComprehensivePseudoUsers() {
    console.log('\n👥 Generating comprehensive pseudo-users from matched reviews...');
    
    // マッチした動画のレビューのみ抽出
    const matchedReviews = this.reviewData.filter(review => 
      this.matchedIds.has(review.content_id)
    );
    
    console.log(`📊 Processing ${matchedReviews.length.toLocaleString()} matched reviews...`);
    
    // レビュワーごとのユーザーデータ生成
    const pseudoUsers = {};
    
    matchedReviews.forEach(review => {
      const reviewer = review.reviewer_name || 'Unknown';
      const rating = review.rating || 0;
      const contentId = review.content_id;
      
      if (!pseudoUsers[reviewer]) {
        pseudoUsers[reviewer] = {
          name: reviewer,
          likes: [],
          skips: [],
          neutral: [],
          totalReviews: 0,
          ratings: []
        };
      }
      
      pseudoUsers[reviewer].totalReviews++;
      pseudoUsers[reviewer].ratings.push(rating);
      
      // 評価基準での分類
      if (rating >= 4.0) {
        pseudoUsers[reviewer].likes.push(contentId);
      } else if (rating <= 3.0) {
        pseudoUsers[reviewer].skips.push(contentId);
      } else {
        pseudoUsers[reviewer].neutral.push(contentId);
      }
    });
    
    // 統計計算
    const userStats = {
      totalUsers: Object.keys(pseudoUsers).length,
      totalLikes: 0,
      totalSkips: 0,
      totalNeutral: 0,
      avgLikesPerUser: 0,
      avgSkipsPerUser: 0,
      likeRate: 0,
      validUsers: 0 // 最低限のアクション数を持つユーザー
    };
    
    // 各ユーザーの統計計算
    const validPseudoUsers = [];
    Object.values(pseudoUsers).forEach(user => {
      const totalActions = user.likes.length + user.skips.length;
      
      if (totalActions >= 3) { // 最低3アクションのユーザーのみ有効
        user.avgRating = user.ratings.reduce((a, b) => a + b, 0) / user.ratings.length;
        user.likeRate = user.likes.length / totalActions * 100;
        validPseudoUsers.push(user);
        
        userStats.totalLikes += user.likes.length;
        userStats.totalSkips += user.skips.length;
        userStats.totalNeutral += user.neutral.length;
        userStats.validUsers++;
      }
    });
    
    userStats.avgLikesPerUser = userStats.validUsers > 0 ? userStats.totalLikes / userStats.validUsers : 0;
    userStats.avgSkipsPerUser = userStats.validUsers > 0 ? userStats.totalSkips / userStats.validUsers : 0;
    userStats.likeRate = (userStats.totalLikes + userStats.totalSkips) > 0 ? 
      (userStats.totalLikes / (userStats.totalLikes + userStats.totalSkips) * 100) : 0;
    
    this.stats.totalUserActions = userStats.totalLikes + userStats.totalSkips;
    
    console.log('📊 Comprehensive Pseudo-User Results:');
    console.log(`   👥 Total users generated: ${userStats.totalUsers}`);
    console.log(`   ✅ Valid users (3+ actions): ${userStats.validUsers}`);
    console.log(`   ❤️  Total likes (4.0+): ${userStats.totalLikes.toLocaleString()}`);
    console.log(`   ⏭️  Total skips (3.0-): ${userStats.totalSkips.toLocaleString()}`);
    console.log(`   ➖ Neutral (3.1-3.9): ${userStats.totalNeutral.toLocaleString()}`);
    console.log(`   📈 Like rate: ${userStats.likeRate.toFixed(1)}%`);
    console.log(`   📊 Avg likes per user: ${userStats.avgLikesPerUser.toFixed(1)}`);
    console.log(`   📊 Avg skips per user: ${userStats.avgSkipsPerUser.toFixed(1)}\n`);
    
    // トップユーザー表示
    console.log('👤 Top Valid Pseudo-Users (by total actions):');
    validPseudoUsers
      .sort((a, b) => (b.likes.length + b.skips.length) - (a.likes.length + a.skips.length))
      .slice(0, 10)
      .forEach((user, i) => {
        const totalActions = user.likes.length + user.skips.length;
        console.log(`   ${i+1}. ${user.name}: ${user.likes.length}L/${user.skips.length}S (${totalActions} actions, avg ${user.avgRating.toFixed(1)}★)`);
      });
    
    // データ保存
    const outputData = {
      users: validPseudoUsers,
      stats: userStats,
      metadata: {
        generatedAt: new Date().toISOString(),
        sourceReviews: matchedReviews.length,
        linkingRate: this.stats.linkingRate,
        parameters: {
          likeThreshold: 4.0,
          skipThreshold: 3.0,
          minActionsPerUser: 3
        }
      }
    };
    
    fs.writeFileSync('data/comprehensive_pseudo_users.json', JSON.stringify(outputData, null, 2));
    console.log(`\n💾 Comprehensive pseudo-users saved to: data/comprehensive_pseudo_users.json`);
    
    return { validPseudoUsers, userStats };
  }

  async run() {
    console.log('🔗 Accurate Content ID Linking & Comprehensive Pseudo-User Generation');
    console.log('=====================================================================\n');
    
    try {
      // データ読み込み
      await this.loadReviewData();
      await this.getAccurateApiVideoCount();
      
      // 直接リンキング
      await this.performDirectLinking();
      
      // 包括的疑似ユーザー生成
      const pseudoUserResults = await this.generateComprehensivePseudoUsers();
      
      // 最終サマリー
      console.log('\n🎯 Final Linking & Training Data Summary:');
      console.log('==========================================');
      console.log(`📊 Review Data: ${this.stats.totalReviews.toLocaleString()} total reviews`);
      console.log(`🎥 API Videos: ${this.stats.totalApiVideos.toLocaleString()} total videos`);
      console.log(`🔗 Linking Success: ${this.stats.matchedVideos.toLocaleString()} videos (${this.stats.linkingRate.toFixed(1)}%)`);
      console.log(`👥 Valid Pseudo-Users: ${pseudoUserResults.userStats.validUsers} users`);
      console.log(`📈 Training Actions: ${this.stats.totalUserActions.toLocaleString()} user actions`);
      console.log(`🎯 ML Training Ready: ${this.stats.totalUserActions >= 1000 ? '✅ YES' : '⚠️ INSUFFICIENT'} (${this.stats.totalUserActions >= 1000 ? 'sufficient' : 'need more'} data)`);
      
    } catch (error) {
      console.error('❌ Accurate linking failed:', error.message);
      throw error;
    }
  }
}

async function main() {
  const linker = new AccurateContentLinker();
  await linker.run();
}

main().catch(console.error);
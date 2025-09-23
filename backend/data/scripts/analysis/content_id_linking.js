const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');

// Supabase設定
const supabaseUrl = 'http://127.0.0.1:54321';
const serviceRoleKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, serviceRoleKey);

class ContentIdLinker {
  constructor() {
    this.reviewData = null;
    this.apiVideoIds = new Set();
    this.linkingStats = {
      totalReviews: 0,
      uniqueContentIds: 0,
      matchedVideos: 0,
      unmatchedReviews: 0,
      linkingRate: 0
    };
  }

  async loadReviewData() {
    console.log('📋 Loading review data...');
    
    // レビューデータファイルを検索
    const reviewFiles = [
      'data_processing/processed_data/integrated_reviews.json',
      'data_processing/processed_data/cleaned_reviews.json',
      'scraped_data/batch_review_data.json',
      'scraped_data/review_data.json',
      'data/batch_review_data.json',
      'data/review_data.json'
    ];
    
    let reviewFile = null;
    for (const file of reviewFiles) {
      if (fs.existsSync(file)) {
        reviewFile = file;
        break;
      }
    }
    
    if (!reviewFile) {
      throw new Error('Review data file not found. Expected locations: ' + reviewFiles.join(', '));
    }
    
    console.log(`📂 Loading from: ${reviewFile}`);
    const rawData = fs.readFileSync(reviewFile, 'utf8');
    this.reviewData = JSON.parse(rawData);
    
    // 統計計算
    this.linkingStats.totalReviews = this.reviewData.length;
    const uniqueIds = new Set(this.reviewData.map(r => r.content_id));
    this.linkingStats.uniqueContentIds = uniqueIds.size;
    
    console.log(`✅ Loaded ${this.linkingStats.totalReviews.toLocaleString()} reviews`);
    console.log(`📊 Unique Content IDs: ${this.linkingStats.uniqueContentIds.toLocaleString()}\n`);
    
    return this.reviewData;
  }

  async loadApiVideoIds() {
    console.log('🎥 Loading API video IDs from database...');
    
    // データベースから全DMM動画のexternal_idを取得
    const { data: videos, error } = await supabase
      .from('videos')
      .select('external_id')
      .eq('source', 'dmm');
    
    if (error) throw error;
    
    this.apiVideoIds.clear();
    videos.forEach(video => {
      this.apiVideoIds.add(video.external_id);
    });
    
    console.log(`✅ Loaded ${this.apiVideoIds.size.toLocaleString()} API video IDs\n`);
    return this.apiVideoIds;
  }

  async performLinking() {
    console.log('🔗 Performing Content ID linking analysis...');
    
    const matchedVideos = new Set();
    const unmatchedReviews = [];
    const linkingDetails = {
      matchedByContentId: 0,
      totalReviewsForMatched: 0
    };
    
    // 各レビューのcontent_idがAPI動画データにあるかチェック
    for (const review of this.reviewData) {
      const contentId = review.content_id;
      
      if (this.apiVideoIds.has(contentId)) {
        matchedVideos.add(contentId);
        linkingDetails.matchedByContentId++;
        linkingDetails.totalReviewsForMatched++;
      } else {
        unmatchedReviews.push({
          content_id: contentId,
          title: review.title,
          rating: review.rating,
          reviewer: review.reviewer_name
        });
      }
    }
    
    // 統計更新
    this.linkingStats.matchedVideos = matchedVideos.size;
    this.linkingStats.unmatchedReviews = unmatchedReviews.length;
    this.linkingStats.linkingRate = (matchedVideos.size / this.linkingStats.uniqueContentIds * 100);
    
    console.log('📊 Linking Analysis Results:');
    console.log(`   ✅ Matched videos: ${matchedVideos.size.toLocaleString()} (${this.linkingStats.linkingRate.toFixed(1)}%)`);
    console.log(`   ❌ Unmatched reviews: ${unmatchedReviews.length.toLocaleString()} unique content IDs`);
    console.log(`   📈 Total reviews for matched videos: ${linkingDetails.totalReviewsForMatched.toLocaleString()}\n`);
    
    return {
      matchedVideos: Array.from(matchedVideos),
      unmatchedReviews: unmatchedReviews.slice(0, 20), // サンプル20件
      linkingDetails
    };
  }

  async analyzeMatchedContent() {
    console.log('🔍 Analyzing matched content characteristics...');
    
    const matchedReviews = this.reviewData.filter(review => 
      this.apiVideoIds.has(review.content_id)
    );
    
    // 評価分布
    const ratingDistribution = {};
    matchedReviews.forEach(review => {
      const rating = review.rating || 0;
      ratingDistribution[rating] = (ratingDistribution[rating] || 0) + 1;
    });
    
    // レビュワー分析
    const reviewerStats = {};
    matchedReviews.forEach(review => {
      const reviewer = review.reviewer_name || 'Unknown';
      if (!reviewerStats[reviewer]) {
        reviewerStats[reviewer] = {
          count: 0,
          ratings: [],
          avgRating: 0
        };
      }
      reviewerStats[reviewer].count++;
      reviewerStats[reviewer].ratings.push(review.rating || 0);
    });
    
    // 平均評価計算
    Object.keys(reviewerStats).forEach(reviewer => {
      const stats = reviewerStats[reviewer];
      stats.avgRating = stats.ratings.reduce((a, b) => a + b, 0) / stats.ratings.length;
    });
    
    console.log('📊 Matched Content Analysis:');
    console.log('   Rating Distribution:');
    Object.entries(ratingDistribution)
      .sort(([a], [b]) => b - a)
      .forEach(([rating, count]) => {
        console.log(`     ${rating}★: ${count} reviews (${(count/matchedReviews.length*100).toFixed(1)}%)`);
      });
    
    console.log('\n   Top Reviewers (matched content):');
    Object.entries(reviewerStats)
      .sort(([,a], [,b]) => b.count - a.count)
      .slice(0, 10)
      .forEach(([reviewer, stats]) => {
        console.log(`     ${reviewer}: ${stats.count} reviews, avg ${stats.avgRating.toFixed(1)}★`);
      });
    
    return {
      matchedReviews: matchedReviews.length,
      ratingDistribution,
      reviewerStats
    };
  }

  async generatePseudoUsers() {
    console.log('\n👥 Generating pseudo-users from matched reviews (4.0+ = Like, 3.0- = Skip)...');
    
    const matchedReviews = this.reviewData.filter(review => 
      this.apiVideoIds.has(review.content_id)
    );
    
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
          totalReviews: 0,
          avgRating: 0
        };
      }
      
      pseudoUsers[reviewer].totalReviews++;
      
      // 4.0以上はLike、3.0以下はSkipとして分類
      if (rating >= 4.0) {
        pseudoUsers[reviewer].likes.push(contentId);
      } else if (rating <= 3.0) {
        pseudoUsers[reviewer].skips.push(contentId);
      }
      // 3.1-3.9は中間評価として除外
    });
    
    // 平均評価計算と統計
    const userStats = {
      totalUsers: Object.keys(pseudoUsers).length,
      totalLikes: 0,
      totalSkips: 0,
      avgLikesPerUser: 0,
      avgSkipsPerUser: 0,
      likeRate: 0
    };
    
    Object.keys(pseudoUsers).forEach(reviewer => {
      const user = pseudoUsers[reviewer];
      const totalActions = user.likes.length + user.skips.length;
      user.avgRating = totalActions > 0 ? 
        (user.likes.length * 4.5 + user.skips.length * 2.5) / totalActions : 0;
      
      userStats.totalLikes += user.likes.length;
      userStats.totalSkips += user.skips.length;
    });
    
    userStats.avgLikesPerUser = userStats.totalLikes / userStats.totalUsers;
    userStats.avgSkipsPerUser = userStats.totalSkips / userStats.totalUsers;
    userStats.likeRate = (userStats.totalLikes / (userStats.totalLikes + userStats.totalSkips) * 100);
    
    console.log('📊 Pseudo-User Generation Results:');
    console.log(`   👥 Generated users: ${userStats.totalUsers}`);
    console.log(`   ❤️  Total likes (4.0+): ${userStats.totalLikes.toLocaleString()}`);
    console.log(`   ⏭️  Total skips (3.0-): ${userStats.totalSkips.toLocaleString()}`);
    console.log(`   📈 Like rate: ${userStats.likeRate.toFixed(1)}%`);
    console.log(`   📊 Avg likes per user: ${userStats.avgLikesPerUser.toFixed(1)}`);
    console.log(`   📊 Avg skips per user: ${userStats.avgSkipsPerUser.toFixed(1)}\n`);
    
    // トップユーザーサンプル表示
    console.log('👤 Top Pseudo-Users (by total actions):');
    Object.values(pseudoUsers)
      .sort((a, b) => (b.likes.length + b.skips.length) - (a.likes.length + a.skips.length))
      .slice(0, 10)
      .forEach((user, i) => {
        console.log(`   ${i+1}. ${user.name}: ${user.likes.length} likes, ${user.skips.length} skips (${user.totalReviews} reviews)`);
      });
    
    // Save pseudo-users data
    const outputFile = 'data/pseudo_users_from_reviews.json';
    fs.writeFileSync(outputFile, JSON.stringify({
      users: Object.values(pseudoUsers),
      stats: userStats,
      generatedAt: new Date().toISOString(),
      parameters: {
        likeThreshold: 4.0,
        skipThreshold: 3.0,
        sourceReviews: matchedReviews.length
      }
    }, null, 2));
    
    console.log(`\n💾 Pseudo-users data saved to: ${outputFile}`);
    
    return { pseudoUsers: Object.values(pseudoUsers), userStats };
  }

  async run() {
    console.log('🔗 Content ID Linking and Pseudo-User Generation');
    console.log('================================================\n');
    
    try {
      // Step 1: Load data
      await this.loadReviewData();
      await this.loadApiVideoIds();
      
      // Step 2: Perform linking analysis
      const linkingResults = await this.performLinking();
      
      // Step 3: Analyze matched content
      const contentAnalysis = await this.analyzeMatchedContent();
      
      // Step 4: Generate pseudo-users
      const pseudoUserResults = await this.generatePseudoUsers();
      
      // Final summary
      console.log('\n🎯 Content ID Linking Summary:');
      console.log('==============================');
      console.log(`📊 Review Data: ${this.linkingStats.totalReviews.toLocaleString()} total reviews`);
      console.log(`🎥 API Videos: ${this.apiVideoIds.size.toLocaleString()} total videos`);
      console.log(`🔗 Linking Rate: ${this.linkingStats.linkingRate.toFixed(1)}% success`);
      console.log(`👥 Generated Users: ${pseudoUserResults.userStats.totalUsers} pseudo-users`);
      console.log(`📈 Training Data: ${pseudoUserResults.userStats.totalLikes + pseudoUserResults.userStats.totalSkips} user actions`);
      
      return {
        linkingStats: this.linkingStats,
        linkingResults,
        contentAnalysis,
        pseudoUserResults
      };
      
    } catch (error) {
      console.error('❌ Content ID linking failed:', error.message);
      throw error;
    }
  }
}

// 実行
async function main() {
  const linker = new ContentIdLinker();
  await linker.run();
}

main().catch(console.error);
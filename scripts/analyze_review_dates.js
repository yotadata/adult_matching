const fs = require('fs');
const path = require('path');

async function analyzeReviewDates() {
  console.log('📅 Review Date Range Analysis');
  console.log('==============================\n');
  
  try {
    // Read integrated reviews data
    const reviewsPath = path.join(__dirname, '../data_processing/processed_data/integrated_reviews.json');
    const rawData = fs.readFileSync(reviewsPath, 'utf8');
    const reviews = JSON.parse(rawData);
    
    console.log(`📊 Total reviews loaded: ${reviews.length.toLocaleString()}`);
    
    // Extract all write_dates and convert to Date objects
    const dates = reviews
      .map(review => review.write_date)
      .filter(date => date && date.trim() !== '')
      .map(dateStr => new Date(dateStr))
      .filter(date => !isNaN(date.getTime())); // Filter out invalid dates
    
    if (dates.length === 0) {
      console.log('❌ No valid dates found in review data');
      return;
    }
    
    // Sort dates
    dates.sort((a, b) => a - b);
    
    const oldestDate = dates[0];
    const newestDate = dates[dates.length - 1];
    
    console.log(`📅 Date Range Analysis:`);
    console.log(`   📍 Oldest review: ${oldestDate.toISOString().split('T')[0]} (${oldestDate.toLocaleDateString()})`);
    console.log(`   📍 Newest review: ${newestDate.toISOString().split('T')[0]} (${newestDate.toLocaleDateString()})`);
    console.log(`   📏 Date span: ${Math.ceil((newestDate - oldestDate) / (1000 * 60 * 60 * 24))} days`);
    
    // Year-by-year breakdown
    const yearCounts = {};
    dates.forEach(date => {
      const year = date.getFullYear();
      yearCounts[year] = (yearCounts[year] || 0) + 1;
    });
    
    console.log(`\n📈 Reviews by Year:`);
    Object.entries(yearCounts)
      .sort(([a], [b]) => parseInt(a) - parseInt(b))
      .forEach(([year, count]) => {
        const percentage = ((count / dates.length) * 100).toFixed(1);
        console.log(`   ${year}: ${count.toLocaleString()} reviews (${percentage}%)`);
      });
    
    // Monthly breakdown for recent years
    const monthCounts = {};
    dates.forEach(date => {
      if (date.getFullYear() >= 2020) {
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        monthCounts[monthKey] = (monthCounts[monthKey] || 0) + 1;
      }
    });
    
    console.log(`\n📅 Recent Monthly Activity (2020+):`);
    Object.entries(monthCounts)
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(-12) // Show last 12 months
      .forEach(([month, count]) => {
        console.log(`   ${month}: ${count.toLocaleString()} reviews`);
      });
    
    // Content ID analysis for DMM API targeting
    const contentIds = [...new Set(reviews.map(r => r.content_id).filter(id => id))];
    console.log(`\n🎯 Target Content IDs for DMM API:`);
    console.log(`   📊 Unique content IDs: ${contentIds.length.toLocaleString()}`);
    console.log(`   📍 Target date range: ${oldestDate.toISOString().split('T')[0]} to ${newestDate.toISOString().split('T')[0]}`);
    
    // Sample content IDs for verification
    console.log(`\n🔍 Sample Content IDs:`);
    const sampleIds = contentIds.slice(0, 10);
    sampleIds.forEach((id, index) => {
      console.log(`   ${index + 1}. ${id}`);
    });
    
    // Estimate DMM API requirements
    const targetStartDate = oldestDate.toISOString().split('T')[0];
    const daysToRetrieve = Math.ceil((new Date() - oldestDate) / (1000 * 60 * 60 * 24));
    const estimatedVideos = Math.min(50000, daysToRetrieve * 10); // Rough estimate: 10 videos per day
    const estimatedPages = Math.ceil(estimatedVideos / 100);
    
    console.log(`\n🎯 DMM API Retrieval Strategy:`);
    console.log(`   📅 Target start date: ${targetStartDate}`);
    console.log(`   📏 Days to retrieve: ${daysToRetrieve.toLocaleString()}`);
    console.log(`   🎬 Estimated videos: ${estimatedVideos.toLocaleString()}`);
    console.log(`   📄 Estimated pages: ${estimatedPages.toLocaleString()}`);
    console.log(`   ⏱️  Estimated time: ${(estimatedPages / 60).toFixed(1)} hours (1 call/sec)`);
    
    return {
      oldestDate: targetStartDate,
      newestDate: newestDate.toISOString().split('T')[0],
      totalContentIds: contentIds.length,
      estimatedVideos,
      estimatedPages,
      dateRange: daysToRetrieve
    };
    
  } catch (error) {
    console.error('❌ Error analyzing review dates:', error.message);
  }
}

// Execute analysis
analyzeReviewDates().catch(console.error);
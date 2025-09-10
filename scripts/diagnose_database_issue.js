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

class DatabaseDiagnostic {
  constructor() {
    this.testResults = [];
  }

  async runComprehensiveDiagnostic() {
    console.log('🔬 Database Issue Diagnostic Tool');
    console.log('==================================\n');

    try {
      // Test 1: データベース接続テスト
      await this.testDatabaseConnection();
      
      // Test 2: トランザクション機能テスト
      await this.testTransactionBehavior();
      
      // Test 3: 挿入操作の詳細テスト
      await this.testInsertOperation();
      
      // Test 4: 既存データとの競合テスト
      await this.testDataConflicts();
      
      // Test 5: データベース制約テスト
      await this.testDatabaseConstraints();

      // Final Results
      console.log('\n🎯 Diagnostic Results Summary:');
      console.log('===============================');
      this.testResults.forEach((result, i) => {
        console.log(`${i+1}. ${result.test}: ${result.status}`);
        if (result.details) {
          console.log(`   Details: ${result.details}`);
        }
      });

    } catch (error) {
      console.error('❌ Diagnostic failed:', error.message);
      throw error;
    }
  }

  async testDatabaseConnection() {
    console.log('📋 Test 1: Database Connection Test');
    
    try {
      // Basic connection test
      const { data: connectionTest, error: connError } = await supabase
        .from('videos')
        .select('count', { count: 'exact' })
        .eq('source', 'dmm');

      if (connError) {
        throw connError;
      }

      console.log(`   ✅ Database connected successfully`);
      console.log(`   📊 Current DMM video count: ${connectionTest[0].count}`);
      
      this.testResults.push({
        test: 'Database Connection',
        status: '✅ PASSED',
        details: `${connectionTest[0].count} records found`
      });

    } catch (error) {
      console.log(`   ❌ Database connection failed: ${error.message}`);
      this.testResults.push({
        test: 'Database Connection',
        status: '❌ FAILED',
        details: error.message
      });
    }
    console.log('');
  }

  async testTransactionBehavior() {
    console.log('📋 Test 2: Transaction Behavior Test');
    
    try {
      // Create a test record with transaction
      const testId = `test_transaction_${Date.now()}`;
      
      console.log(`   🔄 Creating test record: ${testId}`);
      
      const { data: insertResult, error: insertError } = await supabase
        .from('videos')
        .insert({
          external_id: testId,
          title: 'Transaction Test Record',
          description: 'Test for transaction behavior',
          source: 'dmm',
          published_at: new Date().toISOString()
        })
        .select('id, external_id');

      if (insertError) {
        throw insertError;
      }

      console.log(`   ✅ Insert successful: ID ${insertResult[0].id}`);

      // Immediate verification
      const { data: verifyResult, error: verifyError } = await supabase
        .from('videos')
        .select('id, external_id, title')
        .eq('external_id', testId)
        .single();

      if (verifyError) {
        throw new Error(`Verification failed: ${verifyError.message}`);
      }

      console.log(`   ✅ Immediate verification successful: ${verifyResult.title}`);

      // Wait and verify persistence
      console.log('   ⏱️  Waiting 2 seconds for persistence check...');
      await new Promise(resolve => setTimeout(resolve, 2000));

      const { data: persistResult, error: persistError } = await supabase
        .from('videos')
        .select('id, external_id, title')
        .eq('external_id', testId)
        .single();

      if (persistError) {
        throw new Error(`Persistence verification failed: ${persistError.message}`);
      }

      console.log(`   ✅ Persistence verification successful: ${persistResult.title}`);

      // Clean up test record
      const { error: deleteError } = await supabase
        .from('videos')
        .delete()
        .eq('external_id', testId);

      if (deleteError) {
        console.log(`   ⚠️  Cleanup warning: ${deleteError.message}`);
      } else {
        console.log(`   🗑️  Test record cleaned up successfully`);
      }

      this.testResults.push({
        test: 'Transaction Behavior',
        status: '✅ PASSED',
        details: 'Insert, verify, and persist all successful'
      });

    } catch (error) {
      console.log(`   ❌ Transaction test failed: ${error.message}`);
      this.testResults.push({
        test: 'Transaction Behavior',
        status: '❌ FAILED',
        details: error.message
      });
    }
    console.log('');
  }

  async testInsertOperation() {
    console.log('📋 Test 3: Insert Operation Detailed Test');
    
    try {
      // Fetch a fresh item from DMM API
      const { items } = await this.fetchSingleDmmItem();
      
      if (!items || items.length === 0) {
        throw new Error('No items fetched from DMM API');
      }

      const testItem = items[0];
      console.log(`   🎯 Testing with fresh API item: ${testItem.content_id}`);

      // Check if already exists
      const { data: existingCheck } = await supabase
        .from('videos')
        .select('id')
        .eq('external_id', testItem.content_id)
        .eq('source', 'dmm')
        .single();

      if (existingCheck) {
        console.log(`   ℹ️  Item already exists in database, using for conflict test`);
        
        this.testResults.push({
          test: 'Insert Operation',
          status: '🔄 SKIPPED',
          details: 'Item already exists - no new insertion possible'
        });
        return;
      }

      // Prepare video data exactly as in the main script
      const videoData = {
        external_id: testItem.content_id,
        title: testItem.title,
        description: testItem.description || '',
        duration_seconds: null,
        thumbnail_url: testItem.imageURL?.large || testItem.imageURL?.medium || '',
        preview_video_url: testItem.sampleMovieURL?.size_720_480 || testItem.sampleMovieURL?.size_560_360 || '',
        distribution_code: testItem.content_id,
        maker_code: null,
        director: testItem.iteminfo?.director?.[0]?.name || '',
        series: testItem.iteminfo?.series?.[0]?.name || '',
        maker: testItem.iteminfo?.maker?.[0]?.name || '',
        label: testItem.iteminfo?.label?.[0]?.name || '',
        genre: testItem.iteminfo?.genre?.[0]?.name || '',
        price: testItem.prices?.price ? parseInt(String(testItem.prices.price).replace(/[^0-9]/g, '')) || 0 : 0,
        distribution_started_at: testItem.date || null,
        product_released_at: testItem.date || null,
        sample_video_url: testItem.sampleMovieURL?.size_720_480 || '',
        image_urls: testItem.sampleImageURL?.sample_s?.image || [],
        source: 'dmm',
        published_at: testItem.date || new Date().toISOString(),
      };

      console.log(`   📝 Inserting video data...`);
      
      const { data: video, error: videoError } = await supabase
        .from('videos')
        .insert(videoData)
        .select('id, external_id, title')
        .single();

      if (videoError) {
        throw videoError;
      }

      console.log(`   ✅ Video inserted successfully: ${video.external_id} (DB ID: ${video.id})`);

      // Verify immediately
      const { data: verifyInsert } = await supabase
        .from('videos')
        .select('count', { count: 'exact' })
        .eq('source', 'dmm');

      console.log(`   📊 Total DMM videos after insert: ${verifyInsert[0].count}`);

      this.testResults.push({
        test: 'Insert Operation',
        status: '✅ PASSED',
        details: `Successfully inserted and verified ${testItem.content_id}`
      });

    } catch (error) {
      console.log(`   ❌ Insert operation failed: ${error.message}`);
      this.testResults.push({
        test: 'Insert Operation',
        status: '❌ FAILED',
        details: error.message
      });
    }
    console.log('');
  }

  async testDataConflicts() {
    console.log('📋 Test 4: Data Conflicts Test');
    
    try {
      // Get an existing record
      const { data: existingRecord } = await supabase
        .from('videos')
        .select('external_id, title')
        .eq('source', 'dmm')
        .limit(1)
        .single();

      if (!existingRecord) {
        throw new Error('No existing records found for conflict test');
      }

      console.log(`   🎯 Testing conflict with existing record: ${existingRecord.external_id}`);

      // Try to insert duplicate
      const { data: duplicateResult, error: duplicateError } = await supabase
        .from('videos')
        .insert({
          external_id: existingRecord.external_id,
          title: 'Duplicate Test Title',
          description: 'This should conflict',
          source: 'dmm',
          published_at: new Date().toISOString()
        })
        .select('id, external_id');

      if (duplicateError) {
        if (duplicateError.code === '23505') {
          console.log(`   ✅ Unique constraint working properly: ${duplicateError.code}`);
          this.testResults.push({
            test: 'Data Conflicts',
            status: '✅ PASSED',
            details: 'Unique constraints properly enforced'
          });
        } else {
          throw duplicateError;
        }
      } else {
        console.log(`   ❌ Duplicate insertion succeeded unexpectedly`);
        this.testResults.push({
          test: 'Data Conflicts',
          status: '❌ FAILED',
          details: 'Duplicate data was allowed - constraint issue'
        });
      }

    } catch (error) {
      console.log(`   ❌ Data conflicts test failed: ${error.message}`);
      this.testResults.push({
        test: 'Data Conflicts',
        status: '❌ FAILED',
        details: error.message
      });
    }
    console.log('');
  }

  async testDatabaseConstraints() {
    console.log('📋 Test 5: Database Constraints Test');
    
    try {
      // Test required field constraints
      const { data: constraintResult, error: constraintError } = await supabase
        .from('videos')
        .insert({
          // Missing required fields intentionally
          title: 'Constraint Test',
          source: 'dmm'
          // external_id is missing - should fail
        })
        .select('id');

      if (constraintError) {
        console.log(`   ✅ Required field constraints working: ${constraintError.code}`);
        this.testResults.push({
          test: 'Database Constraints',
          status: '✅ PASSED',
          details: 'Required field constraints properly enforced'
        });
      } else {
        console.log(`   ❌ Required field constraint failed - insert succeeded`);
        this.testResults.push({
          test: 'Database Constraints',
          status: '❌ FAILED',
          details: 'Required fields were not enforced'
        });
      }

    } catch (error) {
      console.log(`   ❌ Database constraints test failed: ${error.message}`);
      this.testResults.push({
        test: 'Database Constraints',
        status: '❌ FAILED',
        details: error.message
      });
    }
    console.log('');
  }

  async fetchSingleDmmItem() {
    const params = {
      api_id: DMM_API_CONFIG.api_id,
      affiliate_id: DMM_API_CONFIG.affiliate_id,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      hits: 1,
      offset: Math.floor(Math.random() * 1000) + 1,
      sort: 'date',
      output: 'json'
    };

    const response = await axios.get(DMM_API_CONFIG.baseUrl, { 
      params: params,
      timeout: 30000
    });

    return {
      items: response.data?.result?.items || [],
      totalCount: response.data?.result?.total_count || 0
    };
  }
}

// 実行
async function main() {
  const diagnostic = new DatabaseDiagnostic();
  await diagnostic.runComprehensiveDiagnostic();
}

main().catch(console.error);
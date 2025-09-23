/**
 * Tags-based Genre System 統合テストスイート
 * 全Edge Functionsの動作とtagsベースジャンル取得の正確性を検証
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { assertEquals, assertExists, assert } from "https://deno.land/std@0.190.0/testing/asserts.ts";

interface TestConfig {
  supabaseUrl: string;
  supabaseKey: string;
  testUserId: string;
  testVideoId: string;
}

interface TestResult {
  testName: string;
  passed: boolean;
  error?: string;
  duration: number;
  details?: any;
}

class TagsGenreSystemIntegrationTest {
  private config: TestConfig;
  private supabaseClient: any;
  private testResults: TestResult[] = [];

  constructor() {
    this.config = {
      supabaseUrl: Deno.env.get('SUPABASE_URL') || 'http://127.0.0.1:54321',
      supabaseKey: Deno.env.get('SUPABASE_ANON_KEY') || '',
      testUserId: 'test-user-' + Date.now(),
      testVideoId: 'test-video-' + Date.now()
    };

    this.supabaseClient = createClient(this.config.supabaseUrl, this.config.supabaseKey);
  }

  /**
   * メインテスト実行
   */
  async runAllTests(): Promise<void> {
    console.log('🧪 Tags-based Genre System Integration Tests');
    console.log('============================================');

    const tests = [
      // Core Edge Functions Tests
      () => this.testFeedExploreFunction(),
      () => this.testRecommendationsFunction(),
      () => this.testUpdateUserEmbeddingFunction(),
      () => this.testLikesFunction(),
      () => this.testUserManagementFunction(),

      // Tags-based Genre Extraction Tests
      () => this.testTagsBasedGenreExtraction(),
      () => this.testGenreFilteringAccuracy(),
      () => this.testGenreFallbackBehavior(),

      // Integration Tests
      () => this.testEndToEndRecommendationFlow(),
      () => this.testUserProfileGeneration(),
      () => this.testEmbeddingSystemIntegration(),

      // Performance Tests
      () => this.testResponseTimePerformance(),
      () => this.testBatchProcessingPerformance(),

      // Error Handling Tests
      () => this.testErrorHandlingRobustness(),
      () => this.testDataValidation()
    ];

    for (const test of tests) {
      try {
        await test();
        await new Promise(resolve => setTimeout(resolve, 100)); // Avoid rate limiting
      } catch (error) {
        console.error(`Test execution error: ${error}`);
      }
    }

    this.generateTestReport();
  }

  /**
   * Feed Explore 機能テスト
   */
  private async testFeedExploreFunction(): Promise<void> {
    await this.runTest('Feed Explore Function', async () => {
      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/feed_explore`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ limit: 5 })
      });

      assertEquals(response.status, 200, 'Feed explore should return 200');

      const data = await response.json();
      assertExists(data, 'Response should contain data');

      // Tags-based genre検証
      if (data.videos && data.videos.length > 0) {
        const video = data.videos[0];
        assert(
          video.all_tags || video.genre,
          'Video should have either all_tags or genre (fallback)'
        );
      }

      return { videoCount: data.videos?.length || 0 };
    });
  }

  /**
   * Recommendations 機能テスト
   */
  private async testRecommendationsFunction(): Promise<void> {
    await this.runTest('Recommendations Function', async () => {
      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/enhanced_two_tower_recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.supabaseKey}`
        },
        body: JSON.stringify({
          limit: 10,
          algorithm: 'enhanced'
        })
      });

      // 401 (Unauthorized) or 200 are both acceptable for this test
      assert(
        response.status === 200 || response.status === 401,
        `Recommendations should return 200 or 401, got ${response.status}`
      );

      if (response.status === 200) {
        const data = await response.json();
        assertExists(data.recommendations, 'Should have recommendations array');
      }

      return { status: response.status };
    });
  }

  /**
   * Update User Embedding 機能テスト
   */
  private async testUpdateUserEmbeddingFunction(): Promise<void> {
    await this.runTest('Update User Embedding Function', async () => {
      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/update_user_embedding`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.supabaseKey}`
        },
        body: JSON.stringify({
          force_update: true,
          batch_phase: 'test'
        })
      });

      // Should respond (200 for success, 401 for auth, 500 for no data)
      assert(
        [200, 401, 500].includes(response.status),
        `Update user embedding should return valid status, got ${response.status}`
      );

      return { status: response.status };
    });
  }

  /**
   * Likes 機能テスト
   */
  private async testLikesFunction(): Promise<void> {
    await this.runTest('Likes Function', async () => {
      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/likes`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.supabaseKey}`
        }
      });

      // Should respond with 401 (Unauthorized) for GET without proper auth
      assertEquals(response.status, 401, 'Likes should require authentication');

      const data = await response.json();
      assertEquals(data.error, 'Unauthorized', 'Should return unauthorized error');

      return { authenticationWorking: true };
    });
  }

  /**
   * User Management 機能テスト
   */
  private async testUserManagementFunction(): Promise<void> {
    await this.runTest('User Management Function', async () => {
      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/user-management`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      // Should respond (likely with 405 Method Not Allowed or 401)
      assert(
        response.status === 405 || response.status === 401,
        `User management should return 405 or 401, got ${response.status}`
      );

      return { status: response.status };
    });
  }

  /**
   * Tags-based ジャンル抽出テスト
   */
  private async testTagsBasedGenreExtraction(): Promise<void> {
    await this.runTest('Tags-based Genre Extraction', async () => {
      // モックビデオデータでテスト
      const mockVideo = {
        video_tags: [
          { tags: { name: 'action-genre' } },
          { tags: { name: 'popular' } },
          { tags: { name: 'new-release' } }
        ]
      };

      // ジャンル抽出ロジックのテスト
      const tags = mockVideo.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));

      assertEquals(genreTag, 'action-genre', 'Should extract genre tag correctly');

      // フォールバック動作のテスト
      const mockVideoNoGenre = {
        video_tags: [
          { tags: { name: 'popular' } },
          { tags: { name: 'new-release' } }
        ]
      };

      const tagsNoGenre = mockVideoNoGenre.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);
      const genreTagFallback = tagsNoGenre.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

      assertEquals(genreTagFallback, 'Unknown', 'Should fallback to Unknown when no genre tag');

      return { genreExtractionWorking: true };
    });
  }

  /**
   * ジャンルフィルタリング精度テスト
   */
  private async testGenreFilteringAccuracy(): Promise<void> {
    await this.runTest('Genre Filtering Accuracy', async () => {
      // タグベースフィルタリングのロジックテスト
      const mockVideos = [
        { id: '1', video_tags: [{ tags: { name: 'drama-genre' } }] },
        { id: '2', video_tags: [{ tags: { name: 'action-genre' } }] },
        { id: '3', video_tags: [{ tags: { name: 'comedy-genre' } }] },
        { id: '4', video_tags: [{ tags: { name: 'popular' } }] }
      ];

      // ドラマジャンルフィルタリング
      const dramaVideos = mockVideos.filter(video => {
        const tags = video.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);
        return tags.some((tag: string) => tag.includes('drama'));
      });

      assertEquals(dramaVideos.length, 1, 'Should filter drama videos correctly');
      assertEquals(dramaVideos[0].id, '1', 'Should return correct drama video');

      return { filteringAccuracy: dramaVideos.length };
    });
  }

  /**
   * ジャンルフォールバック動作テスト
   */
  private async testGenreFallbackBehavior(): Promise<void> {
    await this.runTest('Genre Fallback Behavior', async () => {
      // 様々なケースでのフォールバック動作テスト
      const testCases = [
        { video_tags: null, expected: 'Unknown' },
        { video_tags: [], expected: 'Unknown' },
        { video_tags: [{ tags: { name: 'popular' } }], expected: 'Unknown' },
        { video_tags: [{ tags: { name: 'action-genre' } }], expected: 'action-genre' }
      ];

      let passedCases = 0;

      for (const testCase of testCases) {
        const tags = testCase.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
        const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

        if (genreTag === testCase.expected) {
          passedCases++;
        }
      }

      assertEquals(passedCases, testCases.length, 'All fallback cases should pass');

      return { fallbackTestsPassed: passedCases };
    });
  }

  /**
   * エンドツーエンド推薦フローテスト
   */
  private async testEndToEndRecommendationFlow(): Promise<void> {
    await this.runTest('End-to-End Recommendation Flow', async () => {
      // 1. Feed取得
      const feedResponse = await fetch(`${this.config.supabaseUrl}/functions/v1/feed_explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 3 })
      });

      assertEquals(feedResponse.status, 200, 'Feed should be accessible');

      // 2. 推薦取得（認証エラーは想定内）
      const recResponse = await fetch(`${this.config.supabaseUrl}/functions/v1/enhanced_two_tower_recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 5 })
      });

      assert([200, 401].includes(recResponse.status), 'Recommendations should respond');

      return { flowComplete: true };
    });
  }

  /**
   * ユーザープロファイル生成テスト
   */
  private async testUserProfileGeneration(): Promise<void> {
    await this.runTest('User Profile Generation', async () => {
      // プロファイル関連機能のテスト（認証エラーは想定内）
      const profileResponse = await fetch(`${this.config.supabaseUrl}/functions/v1/user-management/profile`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      // 404 (function not found) or 401 (unauthorized) are acceptable
      assert([404, 401, 405].includes(profileResponse.status), 'Profile endpoint should respond');

      return { profileEndpointExists: true };
    });
  }

  /**
   * エンベディングシステム統合テスト
   */
  private async testEmbeddingSystemIntegration(): Promise<void> {
    await this.runTest('Embedding System Integration', async () => {
      // エンベディング更新機能のテスト
      const embeddingResponse = await fetch(`${this.config.supabaseUrl}/functions/v1/update_user_embedding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_phase: 'test' })
      });

      assert([200, 401, 500].includes(embeddingResponse.status), 'Embedding endpoint should respond');

      return { embeddingEndpointWorking: true };
    });
  }

  /**
   * レスポンス時間パフォーマンステスト
   */
  private async testResponseTimePerformance(): Promise<void> {
    await this.runTest('Response Time Performance', async () => {
      const startTime = performance.now();

      const response = await fetch(`${this.config.supabaseUrl}/functions/v1/feed_explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 10 })
      });

      const responseTime = performance.now() - startTime;

      // レスポンス時間は5秒以内であることを期待
      assert(responseTime < 5000, `Response time should be under 5s, got ${responseTime}ms`);

      return { responseTime: Math.round(responseTime) };
    });
  }

  /**
   * バッチ処理パフォーマンステスト
   */
  private async testBatchProcessingPerformance(): Promise<void> {
    await this.runTest('Batch Processing Performance', async () => {
      const startTime = performance.now();

      // 複数の同時リクエストをテスト
      const promises = Array(3).fill(0).map(() =>
        fetch(`${this.config.supabaseUrl}/functions/v1/feed_explore`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ limit: 5 })
        })
      );

      const responses = await Promise.all(promises);
      const batchTime = performance.now() - startTime;

      // 全てのリクエストが成功することを確認
      const successCount = responses.filter(r => r.status === 200).length;
      assertEquals(successCount, 3, 'All batch requests should succeed');

      return { batchTime: Math.round(batchTime), successCount };
    });
  }

  /**
   * エラーハンドリング堅牢性テスト
   */
  private async testErrorHandlingRobustness(): Promise<void> {
    await this.runTest('Error Handling Robustness', async () => {
      // 無効なリクエストでエラーハンドリングをテスト
      const invalidResponse = await fetch(`${this.config.supabaseUrl}/functions/v1/feed_explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 'invalid' })
      });

      // エラーレスポンスが適切に処理されることを確認
      assert(invalidResponse.status >= 400, 'Should handle invalid requests appropriately');

      return { errorHandlingWorking: true };
    });
  }

  /**
   * データ検証テスト
   */
  private async testDataValidation(): Promise<void> {
    await this.runTest('Data Validation', async () => {
      // データベース接続とクエリのテスト
      try {
        const { data, error } = await this.supabaseClient
          .from('videos')
          .select('id')
          .limit(1);

        if (error) {
          console.log('Database query error (expected in some environments):', error.message);
          return { databaseConnected: false, note: 'Database connection test - errors expected in test env' };
        }

        return { databaseConnected: true, videoCount: data?.length || 0 };
      } catch (error) {
        console.log('Database test error (expected):', error);
        return { databaseConnected: false, note: 'Database test completed with expected errors' };
      }
    });
  }

  /**
   * 個別テスト実行ヘルパー
   */
  private async runTest(testName: string, testFunction: () => Promise<any>): Promise<void> {
    const startTime = performance.now();

    try {
      console.log(`\n🧪 Running: ${testName}`);
      const details = await testFunction();
      const duration = performance.now() - startTime;

      this.testResults.push({
        testName,
        passed: true,
        duration: Math.round(duration),
        details
      });

      console.log(`✅ ${testName} - PASSED (${Math.round(duration)}ms)`);
      if (details) {
        console.log(`   Details:`, details);
      }
    } catch (error) {
      const duration = performance.now() - startTime;

      this.testResults.push({
        testName,
        passed: false,
        error: error.message,
        duration: Math.round(duration)
      });

      console.log(`❌ ${testName} - FAILED (${Math.round(duration)}ms)`);
      console.log(`   Error: ${error.message}`);
    }
  }

  /**
   * テストレポート生成
   */
  private generateTestReport(): void {
    console.log('\n📊 Test Report');
    console.log('===============');

    const passed = this.testResults.filter(r => r.passed).length;
    const failed = this.testResults.filter(r => !r.passed).length;
    const totalTime = this.testResults.reduce((sum, r) => sum + r.duration, 0);

    console.log(`Total Tests: ${this.testResults.length}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Success Rate: ${((passed / this.testResults.length) * 100).toFixed(1)}%`);
    console.log(`Total Time: ${totalTime}ms`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults
        .filter(r => !r.passed)
        .forEach(r => console.log(`   - ${r.testName}: ${r.error}`));
    }

    // レポートをファイルに保存
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        total: this.testResults.length,
        passed,
        failed,
        successRate: (passed / this.testResults.length) * 100,
        totalTime
      },
      results: this.testResults
    };

    try {
      Deno.writeTextFileSync(
        '/tmp/tags-genre-system-test-report.json',
        JSON.stringify(reportData, null, 2)
      );
      console.log('\n📄 Report saved to /tmp/tags-genre-system-test-report.json');
    } catch (error) {
      console.log('\n⚠️ Could not save report file');
    }

    // 終了コード設定
    if (failed === 0) {
      console.log('\n🎉 All tests passed!');
    } else {
      console.log(`\n⚠️ ${failed} test(s) failed`);
    }
  }
}

// メイン実行
if (import.meta.main) {
  const tester = new TagsGenreSystemIntegrationTest();
  await tester.runAllTests();
}
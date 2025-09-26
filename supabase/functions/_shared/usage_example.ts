/**
 * 共有ユーティリティ使用例
 * 実際のEdge Functionでの使用方法を示すサンプルコード
 */

import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import {
  createFunctionContext,
  handleOptions,
  validateMethod,
  withErrorHandling,
  finalizeFunctionExecution,
  getSupabaseClientFromRequest,
  validateFeedRequest,
  parseAndValidate,
  responses,
  type FeedValidation
} from './index.ts';

// 使用例: 新しい統一ユーティリティを使った関数の書き方
serve(async (req: Request) => {
  // 1. 関数コンテキストの初期化
  const context = createFunctionContext('example_function', req);

  // 2. OPTIONSリクエストの処理
  if (req.method === 'OPTIONS') {
    return handleOptions();
  }

  // 3. メソッドチェック
  if (!validateMethod(req, ['POST'])) {
    return responses.error('Method not allowed', 405);
  }

  try {
    // 4. リクエストの解析と検証
    const { data, errors } = await parseAndValidate<FeedValidation>(
      req,
      validateFeedRequest
    );

    if (errors) {
      return responses.validationError(errors);
    }

    // 5. データベース接続
    const supabase = getSupabaseClientFromRequest(req);

    // 6. エラーハンドリング付きの処理実行
    const videos = await withErrorHandling(
      context,
      'fetch_videos',
      async () => {
        const { data: videosData, error } = await supabase
          .from('videos')
          .select('*')
          .limit(data!.limit || 20);

        if (error) throw error;
        return videosData;
      }
    );

    // 7. レスポンス生成とファイナライズ
    const response = responses.success(
      videos,
      'Videos fetched successfully',
      performance.now() - context.startTime
    );

    return finalizeFunctionExecution(context, response);

  } catch (error) {
    context.errorTracker.trackError('main_operation', error);
    return responses.internalError(
      'Internal server error',
      performance.now() - context.startTime
    );
  }
});

/**
 * レガシー関数を新しいユーティリティに移行する例
 * Before: 各関数で個別に実装していた内容
 */
/*
// OLD WAY (従来の書き方)
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

if (req.method === 'OPTIONS') {
  return new Response('ok', { headers: corsHeaders });
}

const supabase = createClient(
  Deno.env.get('SUPABASE_URL') ?? '',
  Deno.env.get('SUPABASE_ANON_KEY') ?? '',
  {
    global: {
      headers: { Authorization: req.headers.get('Authorization')! },
    },
  }
);

try {
  const requestData = await req.json();
  // バリデーション処理...
  
  console.log('Processing request...');
  // データベース処理...
  
  return new Response(
    JSON.stringify({ success: true, data: result }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
} catch (error) {
  console.error('Error:', error);
  return new Response(
    JSON.stringify({ error: 'Internal server error' }),
    { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}
*/

/**
 * NEW WAY (新しい統一ユーティリティ使用)
 * - 10行以上のボイラープレートが1〜2行に短縮
 * - 統一されたログ出力
 * - 自動パフォーマンス測定
 * - 標準化されたエラーハンドリング
 * - 統一されたレスポンス形式
 * - 自動バリデーション
 */
import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

// index.ts から serve 関数をインポート
import { serve } from "./index.ts"; // index.ts の serve 関数をインポート

// Supabaseクライアントのモック
const mockSupabaseClient = {
  from: (tableName: string) => ({
    select: (columns: string, options?: any) => {
      const queryState: { offset: number; limit: number; } = { offset: 0, limit: 20 }; // クエリの状態を保持

      const queryBuilder = {
        order: (column: string) => queryBuilder,
        offset: (offset: number) => {
          queryState.offset = offset;
          return queryBuilder;
        },
        limit: (limit: number) => {
          queryState.limit = limit;
          return queryBuilder;
        },
        // 最終的な実行メソッド
        then: async (resolve: any, reject: any) => {
          if (tableName === 'videos' && options?.count === 'exact') {
            resolve({ count: 100, error: null }); // 仮の全件数
          } else {
            const data = Array.from({ length: queryState.limit }, (_, i) => ({
              id: `video_${queryState.offset + i}`,
              title: `Test Video ${queryState.offset + i}`,
              description: 'This is a test video description.',
              sample_video_url: 'http://example.com/sample.mp4',
              genre: ['Action', 'Comedy'],
              external_id: `ext_${queryState.offset + i}`,
              product_url: 'http://example.com/product',
            }));
            resolve({ data, error: null });
          }
        },
      };
      return queryBuilder;
    },
  }),
};

// Deno.env.get のモック
const originalDenoEnvGet = Deno.env.get;
Deno.env.get = (key: string) => {
  if (key === 'SUPABASE_URL') return 'http://mock-supabase-url';
  if (key === 'SUPABASE_ANON_KEY') return 'mock-anon-key';
  return originalDenoEnvGet(key);
};

Deno.test("videos-feed API returns 20 videos", async () => {
  const req = new Request("http://localhost:8000/", { method: "GET" });

  // serve 関数を直接呼び出す
  const res = await serve(req); // serve 関数を直接呼び出す

  assertEquals(res.status, 200);
  const body = await res.json();
  assert(Array.isArray(body));
  assertEquals(body.length, 20);
  assert(body[0].id.startsWith("video_"));
  assert(body[0].title.startsWith("Test Video"));
});

// Deno.env.get のモックを元に戻す
Deno.env.get = originalDenoEnvGet;
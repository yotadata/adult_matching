/**
 * Supabase整合性チェッカー設定ファイル
 * 汎用設計で他のSupabaseプロジェクトでも利用可能
 */

export default {
    // プロジェクトパス設定
    frontendPath: "frontend/src",
    backendPath: "supabase/functions",
    databasePath: "supabase/migrations",
    outputPath: "reports",

    // Supabase接続設定
    supabaseUrl: process.env.SUPABASE_URL,
    supabaseKey: process.env.SUPABASE_ANON_KEY,

    // 解析対象ファイルフィルター
    skipPatterns: [
        ".test.",
        ".spec.",
        "node_modules",
        ".git",
        "dist",
        "build"
    ],
    includePatterns: [
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".sql"
    ],

    // 期待されるEdge Functions
    expectedFunctions: [
        "videos-feed",
        "likes",
        "update_user_embedding",
        "delete_account"
    ],

    // 期待されるRPC関数
    expectedRpcFunctions: [
        "get_videos_feed",
        "get_user_likes",
        "get_user_liked_tags",
        "get_user_liked_performers"
    ],

    // テスト環境設定
    environments: ["development"],
    enableLiveTests: true,
    testTimeout: 15000,
    maxRetries: 3,
    retryDelay: 1000,

    // レポート設定
    generateHtmlReport: true,
    generateMarkdownReport: true,
    generateJsonReport: true,

    // 認証テスト用データ（オプション）
    testData: {
        "videos-feed": {},
        "likes": undefined,
        "update_user_embedding": {
            batch_phase: "test",
            batch_size: 10,
            completed_items: 5
        },
        "delete_account": {}
    },

    // 詳細設定
    verboseLogging: false,
    continueOnError: true,
    parallelExecution: false
};
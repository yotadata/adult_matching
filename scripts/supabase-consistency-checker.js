#!/usr/bin/env node

import { chalk, fs, path } from "zx";
import { SupabaseAnalyzer } from "./utilities/supabase-analyzer.js";
import { EdgeFunctionValidator } from "./utilities/edge-function-validator.js";
import { DatabaseSchemaChecker } from "./utilities/database-schema-checker.js";
import { ApiResponseValidator } from "./utilities/api-response-validator.js";

/**
 * Supabase整合性チェッカー - メインCLI
 * 汎用的なSupabaseプロジェクト整合性検証ツール
 */
class SupabaseConsistencyChecker {
    constructor() {
        this.config = {};
        this.results = {};
        this.allIssues = [];
    }

    /**
     * CLIメイン実行
     */
    async run() {
        console.log(chalk.blue.bold("🚀 Supabase整合性チェッカー v1.0.0"));
        console.log(chalk.gray("汎用Supabaseプロジェクト整合性検証ツール\n"));

        try {
            // 設定読み込み
            await this.loadConfig();

            // 全体解析実行
            await this.performAnalysis();

            // 結果出力
            await this.generateReports();

            // サマリー表示
            this.printFinalSummary();

            // 終了コード決定
            const exitCode = this.allIssues.length > 0 ? 1 : 0;
            process.exit(exitCode);

        } catch (error) {
            console.error(chalk.red("❌ 実行エラー:"), error.message);
            process.exit(1);
        }
    }

    /**
     * 設定読み込み
     */
    async loadConfig() {
        const configPaths = [
            "scripts/config/supabase-consistency.config.js",
            "supabase-consistency.config.js",
            ".supabase-consistency.json"
        ];

        for (const configPath of configPaths) {
            if (await fs.pathExists(configPath)) {
                console.log(chalk.green(`📋 設定ファイル読み込み: ${configPath}`));

                if (configPath.endsWith('.js')) {
                    const { default: config } = await import(path.resolve(configPath));
                    this.config = config;
                } else {
                    const configContent = await fs.readFile(configPath, 'utf8');
                    this.config = JSON.parse(configContent);
                }
                return;
            }
        }

        // デフォルト設定
        this.config = this.getDefaultConfig();
        console.log(chalk.yellow("⚠️  設定ファイルが見つかりません。デフォルト設定を使用します。"));
    }

    /**
     * デフォルト設定取得
     */
    getDefaultConfig() {
        return {
            frontendPath: "frontend/src",
            backendPath: "supabase/functions",
            databasePath: "supabase/migrations",
            outputPath: "reports",
            supabaseUrl: process.env.SUPABASE_URL,
            supabaseKey: process.env.SUPABASE_ANON_KEY,
            skipPatterns: [".test.", ".spec.", "node_modules"],
            includePatterns: [".ts", ".tsx", ".js", ".jsx", ".sql"],
            expectedFunctions: [
                "videos-feed",
                "likes",
                "update_user_embedding",
                "delete_account"
            ],
            expectedRpcFunctions: [
                "get_videos_feed",
                "get_user_likes",
                "get_user_liked_tags",
                "get_user_liked_performers"
            ],
            environments: ["development"],
            enableLiveTests: true,
            generateHtmlReport: true
        };
    }

    /**
     * 全体解析実行
     */
    async performAnalysis() {
        console.log(chalk.blue("🔍 整合性解析開始...\n"));

        // 1. Supabase環境解析
        console.log(chalk.cyan("1️⃣  Supabase環境解析"));
        const analyzer = new SupabaseAnalyzer(this.config);
        this.results.analysis = await analyzer.analyzeSupabaseEnvironment();
        analyzer.checkConsistency();
        this.allIssues.push(...analyzer.issues);

        // 2. Edge Functions検証
        console.log(chalk.cyan("\n2️⃣  Edge Functions検証"));
        const functionValidator = new EdgeFunctionValidator(this.config);
        this.results.edgeFunctions = await functionValidator.validateAllFunctions();
        this.allIssues.push(...functionValidator.issues);

        // 3. データベーススキーマチェック
        console.log(chalk.cyan("\n3️⃣  データベーススキーマチェック"));
        const schemaChecker = new DatabaseSchemaChecker(this.config);
        this.results.databaseSchema = await schemaChecker.checkSchemaConsistency();
        this.allIssues.push(...schemaChecker.issues);

        // 4. API実環境検証（オプション）
        if (this.config.enableLiveTests && this.config.supabaseUrl && this.config.supabaseKey) {
            console.log(chalk.cyan("\n4️⃣  API実環境検証"));
            const apiValidator = new ApiResponseValidator(this.config);
            this.results.apiValidation = await apiValidator.validateApiResponses();
            this.allIssues.push(...apiValidator.issues);
        } else {
            console.log(chalk.yellow("⚠️  API実環境検証をスキップ（設定またはクレデンシャル不足）"));
        }

        console.log(chalk.green("\n✅ 全解析完了"));
    }

    /**
     * レポート生成
     */
    async generateReports() {
        console.log(chalk.blue("\n📊 レポート生成中..."));

        // ディレクトリ作成
        await fs.ensureDir(this.config.outputPath);

        // JSON詳細レポート
        const jsonReport = {
            metadata: {
                timestamp: new Date().toISOString(),
                version: "1.0.0",
                config: this.config
            },
            results: this.results,
            issues: this.allIssues,
            summary: this.generateSummary()
        };

        const jsonPath = path.join(this.config.outputPath, "supabase-consistency-report.json");
        await fs.writeFile(jsonPath, JSON.stringify(jsonReport, null, 2));
        console.log(chalk.green(`📄 JSON詳細レポート: ${jsonPath}`));

        // Markdownサマリーレポート
        const markdownReport = this.generateMarkdownReport(jsonReport);
        const mdPath = path.join(this.config.outputPath, "supabase-consistency-summary.md");
        await fs.writeFile(mdPath, markdownReport);
        console.log(chalk.green(`📝 Markdownサマリー: ${mdPath}`));

        // HTMLレポート（オプション）
        if (this.config.generateHtmlReport) {
            const htmlReport = this.generateHtmlReport(jsonReport);
            const htmlPath = path.join(this.config.outputPath, "supabase-consistency-report.html");
            await fs.writeFile(htmlPath, htmlReport);
            console.log(chalk.green(`🌐 HTMLレポート: ${htmlPath}`));
        }
    }

    /**
     * Markdownレポート生成
     */
    generateMarkdownReport(jsonReport) {
        const { summary, issues } = jsonReport;

        return `# Supabase整合性チェック レポート

## 📊 サマリー

- **実行日時**: ${summary.timestamp}
- **検証項目**: ${summary.totalChecks}項目
- **発見された問題**: ${summary.totalIssues}件
- **成功率**: ${summary.successRate}%

## 🔍 検証結果

### Edge Functions
- 検証対象: ${summary.edgeFunctions.total}個
- 成功: ${summary.edgeFunctions.passed}個
- 失敗: ${summary.edgeFunctions.failed}個

### RPC Functions
- 定義済み関数: ${summary.rpcFunctions.defined}個
- 使用箇所: ${summary.rpcFunctions.usages}箇所
- 整合性: ${summary.rpcFunctions.consistent ? '✅' : '❌'}

### API検証
- テスト済み環境: ${summary.apiValidation.environments}環境
- 実行テスト: ${summary.apiValidation.tests}件
- 成功率: ${summary.apiValidation.successRate}%

## ⚠️ 発見された問題

${issues.length === 0 ? '問題は見つかりませんでした。' : ''}

${issues.map((issue, index) => `
### ${index + 1}. ${issue.type}

**メッセージ**: ${issue.message}

**詳細**: ${issue.details ? JSON.stringify(issue.details, null, 2) : 'なし'}

**発生時刻**: ${issue.timestamp}

---`).join('\n')}

## 🛠️ 推奨アクション

${this.generateRecommendations(issues)}

---

*Generated by Supabase Consistency Checker v1.0.0*
`;
    }

    /**
     * HTMLレポート生成
     */
    generateHtmlReport(jsonReport) {
        const { summary, issues } = jsonReport;

        return `<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supabase整合性チェック レポート</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; border-left: 4px solid #1a73e8; }
        .metric-value { font-size: 2em; font-weight: bold; color: #1a73e8; }
        .metric-label { color: #666; margin-top: 5px; }
        .issue { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .issue.error { background: #f8d7da; border-color: #dc3545; }
        .issue.warning { background: #d1ecf1; border-color: #0dcaf0; }
        .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .status-success { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
        .status-warning { background: #fff3cd; color: #856404; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Supabase整合性チェック レポート</h1>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">${summary.totalChecks}</div>
                <div class="metric-label">検証項目</div>
            </div>
            <div class="metric">
                <div class="metric-value">${summary.totalIssues}</div>
                <div class="metric-label">発見された問題</div>
            </div>
            <div class="metric">
                <div class="metric-value">${summary.successRate}%</div>
                <div class="metric-label">成功率</div>
            </div>
            <div class="metric">
                <div class="metric-value">${summary.timestamp.split('T')[0]}</div>
                <div class="metric-label">実行日</div>
            </div>
        </div>

        <h2>📋 検証結果詳細</h2>

        <h3>Edge Functions</h3>
        <p>検証対象: ${summary.edgeFunctions.total}個 |
           成功: <span class="status-badge status-success">${summary.edgeFunctions.passed}</span> |
           失敗: <span class="status-badge status-error">${summary.edgeFunctions.failed}</span></p>

        <h3>RPC Functions</h3>
        <p>定義済み: ${summary.rpcFunctions.defined}個 |
           使用箇所: ${summary.rpcFunctions.usages}箇所 |
           整合性: <span class="status-badge ${summary.rpcFunctions.consistent ? 'status-success' : 'status-error'}">${summary.rpcFunctions.consistent ? '✅ OK' : '❌ NG'}</span></p>

        <h2>⚠️ 発見された問題 (${issues.length}件)</h2>

        ${issues.length === 0 ? '<p class="status-badge status-success">問題は見つかりませんでした！</p>' : ''}

        ${issues.map((issue, index) => `
            <div class="issue ${issue.type.includes('error') ? 'error' : 'warning'}">
                <h4>${index + 1}. ${issue.type}</h4>
                <p><strong>メッセージ:</strong> ${issue.message}</p>
                ${issue.details ? `<p><strong>詳細:</strong></p><pre>${JSON.stringify(issue.details, null, 2)}</pre>` : ''}
                <p><small><strong>発生時刻:</strong> ${issue.timestamp}</small></p>
            </div>
        `).join('')}

        <h2>🛠️ 推奨アクション</h2>
        <div>${this.generateRecommendations(issues)}</div>

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #666;">
            <em>Generated by Supabase Consistency Checker v1.0.0</em><br>
            <small>${new Date().toLocaleString('ja-JP')}</small>
        </p>
    </div>
</body>
</html>`;
    }

    /**
     * 推奨アクション生成
     */
    generateRecommendations(issues) {
        if (issues.length === 0) {
            return "すべての整合性チェックに合格しています。定期的な再チェックを推奨します。";
        }

        const recommendations = [];
        const issueTypes = [...new Set(issues.map(i => i.type))];

        issueTypes.forEach(type => {
            const count = issues.filter(i => i.type === type).length;

            switch (type) {
                case "missing_edge_function":
                    recommendations.push(`- **Edge Function不足 (${count}件)**: 対応するEdge Functionを実装するか、不要なAPI呼び出しを削除してください。`);
                    break;
                case "missing_rpc_function":
                    recommendations.push(`- **RPC関数不足 (${count}件)**: データベースマイグレーションでRPC関数を定義するか、呼び出し箇所を修正してください。`);
                    break;
                case "response_format":
                    recommendations.push(`- **レスポンス形式不整合 (${count}件)**: Edge Functionsのレスポンス形式を統一してください。`);
                    break;
                case "auth_inconsistency":
                    recommendations.push(`- **認証不整合 (${count}件)**: 認証ヘッダーの設定とRLSポリシーを確認してください。`);
                    break;
                default:
                    recommendations.push(`- **${type} (${count}件)**: 詳細を確認して適切な対応を行ってください。`);
            }
        });

        return recommendations.join('\n');
    }

    /**
     * サマリー生成
     */
    generateSummary() {
        const edgeFunctionResults = this.results.edgeFunctions?.validationResults || [];
        const rpcResults = this.results.databaseSchema || {};
        const apiResults = this.results.apiValidation || {};

        return {
            timestamp: new Date().toISOString(),
            totalChecks: edgeFunctionResults.length + (rpcResults.rpcFunctions?.length || 0),
            totalIssues: this.allIssues.length,
            successRate: this.allIssues.length === 0 ? 100 : Math.round((1 - this.allIssues.length / Math.max(1, edgeFunctionResults.length)) * 100),
            edgeFunctions: {
                total: edgeFunctionResults.length,
                passed: edgeFunctionResults.filter(r => r.issues.length === 0).length,
                failed: edgeFunctionResults.filter(r => r.issues.length > 0).length
            },
            rpcFunctions: {
                defined: rpcResults.rpcFunctions?.length || 0,
                usages: rpcResults.rpcUsage?.length || 0,
                consistent: (rpcResults.issues?.length || 0) === 0
            },
            apiValidation: {
                environments: Object.keys(apiResults.environmentResults || {}).length,
                tests: Object.values(apiResults.environmentResults || {}).reduce((sum, env) => sum + env.summary.tested, 0),
                successRate: this.calculateApiSuccessRate(apiResults)
            }
        };
    }

    calculateApiSuccessRate(apiResults) {
        const envResults = Object.values(apiResults.environmentResults || {});
        if (envResults.length === 0) return 0;

        const totalTests = envResults.reduce((sum, env) => sum + env.summary.tested, 0);
        const passedTests = envResults.reduce((sum, env) => sum + env.summary.passed, 0);

        return totalTests > 0 ? Math.round((passedTests / totalTests) * 100) : 0;
    }

    /**
     * 最終サマリー表示
     */
    printFinalSummary() {
        console.log(chalk.blue.bold("\n🎯 最終サマリー"));
        console.log("=====================================");

        const summary = this.generateSummary();

        console.log(`📊 総検証項目: ${summary.totalChecks}`);
        console.log(`🎯 成功率: ${summary.successRate}%`);

        if (summary.totalIssues === 0) {
            console.log(chalk.green.bold("✅ すべての整合性チェックに合格しました！"));
        } else {
            console.log(chalk.red(`⚠️  ${summary.totalIssues}件の問題が発見されました。`));

            // 問題の種類別集計
            const issuesByType = {};
            this.allIssues.forEach(issue => {
                issuesByType[issue.type] = (issuesByType[issue.type] || 0) + 1;
            });

            console.log(chalk.yellow("\n問題の内訳:"));
            Object.entries(issuesByType).forEach(([type, count]) => {
                console.log(`  - ${type}: ${count}件`);
            });
        }

        console.log(chalk.gray(`\n📁 詳細レポート: ${this.config.outputPath}/`));
        console.log(chalk.gray(`⏰ 実行完了: ${new Date().toLocaleString('ja-JP')}`));
    }
}

// CLI実行
if (import.meta.url === `file://${process.argv[1]}`) {
    const checker = new SupabaseConsistencyChecker();
    checker.run();
}

export { SupabaseConsistencyChecker };
#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

/**
 * 簡易Supabase整合性チェッカー
 * 依存関係なしでフロントエンド・バックエンド整合性をチェック
 */
class SimpleConsistencyChecker {
    constructor() {
        this.issues = [];
        this.results = {
            edgeFunctions: [],
            apiCalls: [],
            rpcFunctions: [],
            typeDefinitions: []
        };
    }

    /**
     * メイン実行
     */
    async run() {
        console.log("🚀 Supabase整合性チェック開始...");
        console.log("=====================================\n");

        try {
            // Edge Functions検出
            await this.findEdgeFunctions();

            // フロントエンドAPI呼び出し検出
            await this.findFrontendApiCalls();

            // RPC関数定義検出
            await this.findRpcFunctions();

            // 型定義検出
            await this.findTypeDefinitions();

            // 整合性チェック実行
            this.performConsistencyChecks();

            // 結果出力
            this.printResults();

        } catch (error) {
            console.error("❌ エラー:", error.message);
        }
    }

    /**
     * Edge Functions検出
     */
    async findEdgeFunctions() {
        console.log("📁 Edge Functions検出中...");

        const functionsPath = "supabase/functions";
        if (!fs.existsSync(functionsPath)) {
            this.addIssue("missing_directory", `Edge Functionsディレクトリが見つかりません: ${functionsPath}`);
            return;
        }

        const entries = fs.readdirSync(functionsPath, { withFileTypes: true });

        for (const entry of entries) {
            if (entry.isDirectory()) {
                const indexPath = path.join(functionsPath, entry.name, "index.ts");
                if (fs.existsSync(indexPath)) {
                    this.results.edgeFunctions.push({
                        name: entry.name,
                        path: indexPath
                    });
                    console.log(`  ✅ ${entry.name}`);
                }
            }
        }

        console.log(`📊 Edge Functions: ${this.results.edgeFunctions.length}個発見\n`);
    }

    /**
     * フロントエンドAPI呼び出し検出
     */
    async findFrontendApiCalls() {
        console.log("🎨 フロントエンドAPI呼び出し検出中...");

        const frontendPath = "frontend/src";
        if (!fs.existsSync(frontendPath)) {
            this.addIssue("missing_directory", `フロントエンドディレクトリが見つかりません: ${frontendPath}`);
            return;
        }

        await this.searchInDirectory(frontendPath, /\.(ts|tsx|js|jsx)$/, (filePath, content) => {
            // supabase.functions.invoke 呼び出し検出
            const invokeMatches = content.matchAll(/supabase\.functions\.invoke\(['"`]([^'"`]+)['"`]/g);
            for (const match of invokeMatches) {
                this.results.apiCalls.push({
                    type: "edge_function",
                    functionName: match[1],
                    file: filePath,
                    line: this.getLineNumber(content, match.index)
                });
            }

            // supabase.rpc 呼び出し検出
            const rpcMatches = content.matchAll(/supabase\.rpc\(['"`]([^'"`]+)['"`]/g);
            for (const match of rpcMatches) {
                this.results.apiCalls.push({
                    type: "rpc_call",
                    functionName: match[1],
                    file: filePath,
                    line: this.getLineNumber(content, match.index)
                });
            }
        });

        const edgeFunctionCalls = this.results.apiCalls.filter(call => call.type === "edge_function");
        const rpcCalls = this.results.apiCalls.filter(call => call.type === "rpc_call");

        console.log(`📊 Edge Function呼び出し: ${edgeFunctionCalls.length}箇所`);
        console.log(`📊 RPC呼び出し: ${rpcCalls.length}箇所\n`);
    }

    /**
     * RPC関数定義検出
     */
    async findRpcFunctions() {
        console.log("🗄️ RPC関数定義検出中...");

        const migrationsPath = "supabase/migrations";
        if (!fs.existsSync(migrationsPath)) {
            this.addIssue("missing_directory", `マイグレーションディレクトリが見つかりません: ${migrationsPath}`);
            return;
        }

        await this.searchInDirectory(migrationsPath, /\.sql$/, (filePath, content) => {
            const functionMatches = content.matchAll(/CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi);
            for (const match of functionMatches) {
                this.results.rpcFunctions.push({
                    name: match[1],
                    file: filePath,
                    line: this.getLineNumber(content, match.index)
                });
            }
        });

        console.log(`📊 RPC関数定義: ${this.results.rpcFunctions.length}個発見\n`);
    }

    /**
     * 型定義検出
     */
    async findTypeDefinitions() {
        console.log("📝 型定義検出中...");

        const frontendPath = "frontend/src";
        if (!fs.existsSync(frontendPath)) return;

        await this.searchInDirectory(frontendPath, /\.(ts|tsx)$/, (filePath, content) => {
            // interface定義検出
            const interfaceMatches = content.matchAll(/interface\s+([a-zA-Z_][a-zA-Z0-9_]*)/g);
            for (const match of interfaceMatches) {
                this.results.typeDefinitions.push({
                    type: "interface",
                    name: match[1],
                    file: filePath,
                    line: this.getLineNumber(content, match.index)
                });
            }

            // type定義検出
            const typeMatches = content.matchAll(/type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=/g);
            for (const match of typeMatches) {
                this.results.typeDefinitions.push({
                    type: "type_alias",
                    name: match[1],
                    file: filePath,
                    line: this.getLineNumber(content, match.index)
                });
            }
        });

        console.log(`📊 型定義: ${this.results.typeDefinitions.length}個発見\n`);
    }

    /**
     * 整合性チェック実行
     */
    performConsistencyChecks() {
        console.log("🔍 整合性チェック実行中...");

        // Edge Function整合性チェック
        this.checkEdgeFunctionConsistency();

        // RPC関数整合性チェック
        this.checkRpcConsistency();

        // 重要な型定義存在チェック
        this.checkImportantTypes();

        console.log(`✅ 整合性チェック完了: ${this.issues.length}件の問題を発見\n`);
    }

    /**
     * Edge Function整合性チェック
     */
    checkEdgeFunctionConsistency() {
        const definedFunctions = this.results.edgeFunctions.map(f => f.name);
        const calledFunctions = this.results.apiCalls
            .filter(call => call.type === "edge_function")
            .map(call => call.functionName);

        // 呼び出されているが定義されていない関数
        const uniqueCalledFunctions = [...new Set(calledFunctions)];
        uniqueCalledFunctions.forEach(funcName => {
            if (!definedFunctions.includes(funcName)) {
                const usages = this.results.apiCalls.filter(
                    call => call.type === "edge_function" && call.functionName === funcName
                );
                usages.forEach(usage => {
                    this.addIssue(
                        "missing_edge_function",
                        `Edge Function「${funcName}」が定義されていません`,
                        { file: usage.file, line: usage.line }
                    );
                });
            }
        });

        // 定義されているが呼び出されていない関数
        definedFunctions.forEach(funcName => {
            if (!calledFunctions.includes(funcName)) {
                this.addIssue(
                    "unused_edge_function",
                    `Edge Function「${funcName}」が使用されていません`,
                    this.results.edgeFunctions.find(f => f.name === funcName)
                );
            }
        });
    }

    /**
     * RPC関数整合性チェック
     */
    checkRpcConsistency() {
        const definedRpcFunctions = this.results.rpcFunctions.map(f => f.name);
        const calledRpcFunctions = this.results.apiCalls
            .filter(call => call.type === "rpc_call")
            .map(call => call.functionName);

        // 呼び出されているが定義されていないRPC関数
        const uniqueCalledRpc = [...new Set(calledRpcFunctions)];
        uniqueCalledRpc.forEach(funcName => {
            if (!definedRpcFunctions.includes(funcName)) {
                const usages = this.results.apiCalls.filter(
                    call => call.type === "rpc_call" && call.functionName === funcName
                );
                usages.forEach(usage => {
                    this.addIssue(
                        "missing_rpc_function",
                        `RPC関数「${funcName}」が定義されていません`,
                        { file: usage.file, line: usage.line }
                    );
                });
            }
        });

        // 期待されるRPC関数の存在確認
        const expectedRpcFunctions = [
            "get_videos_feed",
            "get_user_likes",
            "get_user_liked_tags",
            "get_user_liked_performers"
        ];

        expectedRpcFunctions.forEach(expectedFunc => {
            if (!definedRpcFunctions.includes(expectedFunc)) {
                this.addIssue(
                    "missing_expected_rpc",
                    `期待されるRPC関数「${expectedFunc}」が定義されていません`
                );
            }
        });
    }

    /**
     * 重要な型定義存在チェック
     */
    checkImportantTypes() {
        const definedTypes = this.results.typeDefinitions.map(t => t.name);
        const importantTypes = ["VideoFromApi", "CardData"];

        importantTypes.forEach(typeName => {
            if (!definedTypes.includes(typeName)) {
                this.addIssue(
                    "missing_important_type",
                    `重要な型定義「${typeName}」が見つかりません`
                );
            }
        });
    }

    /**
     * ヘルパーメソッド
     */
    async searchInDirectory(dir, pattern, callback) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);

            if (entry.isDirectory()) {
                await this.searchInDirectory(fullPath, pattern, callback);
            } else if (pattern.test(entry.name)) {
                try {
                    const content = fs.readFileSync(fullPath, 'utf8');
                    callback(fullPath, content);
                } catch (error) {
                    console.log(`⚠️  ファイル読み込みエラー: ${fullPath}`);
                }
            }
        }
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length;
    }

    addIssue(type, message, details = null) {
        this.issues.push({
            type,
            message,
            details,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * 結果出力
     */
    printResults() {
        console.log("📋 整合性チェック結果");
        console.log("=====================================");

        // サマリー
        console.log("📊 サマリー:");
        console.log(`  Edge Functions: ${this.results.edgeFunctions.length}個`);
        console.log(`  API呼び出し: ${this.results.apiCalls.length}箇所`);
        console.log(`  RPC関数定義: ${this.results.rpcFunctions.length}個`);
        console.log(`  型定義: ${this.results.typeDefinitions.length}個`);
        console.log(`  発見された問題: ${this.issues.length}件`);

        // Edge Functions詳細
        console.log("\n🔧 Edge Functions:");
        this.results.edgeFunctions.forEach(func => {
            console.log(`  ✅ ${func.name}`);
        });

        // API呼び出し詳細
        console.log("\n📞 API呼び出し:");
        const groupedCalls = {};
        this.results.apiCalls.forEach(call => {
            const key = `${call.type}:${call.functionName}`;
            if (!groupedCalls[key]) {
                groupedCalls[key] = [];
            }
            groupedCalls[key].push(call);
        });

        Object.entries(groupedCalls).forEach(([key, calls]) => {
            const [type, functionName] = key.split(':');
            const icon = type === "edge_function" ? "⚡" : "🗄️";
            console.log(`  ${icon} ${functionName} (${calls.length}箇所)`);
        });

        // RPC関数詳細
        console.log("\n🗄️ RPC関数:");
        this.results.rpcFunctions.forEach(func => {
            console.log(`  ✅ ${func.name}`);
        });

        // 問題詳細
        if (this.issues.length > 0) {
            console.log("\n⚠️  発見された問題:");
            this.issues.forEach((issue, index) => {
                console.log(`\n${index + 1}. ${issue.type}`);
                console.log(`   ${issue.message}`);
                if (issue.details && issue.details.file) {
                    console.log(`   📁 ${issue.details.file}:${issue.details.line || 'unknown'}`);
                }
            });
        } else {
            console.log("\n✅ 問題は見つかりませんでした！");
        }

        // 推奨アクション
        console.log("\n🛠️ 推奨アクション:");
        if (this.issues.length === 0) {
            console.log("  ✅ すべての整合性チェックに合格しています。");
        } else {
            const actionsByType = {};
            this.issues.forEach(issue => {
                if (!actionsByType[issue.type]) {
                    actionsByType[issue.type] = [];
                }
                actionsByType[issue.type].push(issue);
            });

            Object.entries(actionsByType).forEach(([type, issues]) => {
                switch (type) {
                    case "missing_edge_function":
                        console.log(`  🔧 Edge Function不足 (${issues.length}件): 対応するEdge Functionを実装してください`);
                        break;
                    case "missing_rpc_function":
                        console.log(`  🗄️ RPC関数不足 (${issues.length}件): データベースマイグレーションでRPC関数を定義してください`);
                        break;
                    case "unused_edge_function":
                        console.log(`  🧹 未使用Edge Function (${issues.length}件): 不要な関数を削除するか、使用箇所を追加してください`);
                        break;
                    case "missing_important_type":
                        console.log(`  📝 重要な型定義不足 (${issues.length}件): 必要な型定義を追加してください`);
                        break;
                    default:
                        console.log(`  ⚠️  ${type} (${issues.length}件): 詳細を確認して対応してください`);
                }
            });
        }

        console.log("\n=====================================");
        console.log(`✅ 整合性チェック完了 (${new Date().toLocaleString('ja-JP')})`);
    }
}

// 実行
const checker = new SimpleConsistencyChecker();
checker.run();
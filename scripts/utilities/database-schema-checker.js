import { chalk, fs, path } from "zx";
import { parseFiles } from "@ast-grep/napi";

/**
 * データベーススキーマ整合性チェッカー
 * 汎用PostgreSQL+Supabase検証ツール
 * RPC関数定義とEdge Functions使用の整合性を確認
 */
export class DatabaseSchemaChecker {
    constructor(config = {}) {
        this.config = {
            migrationsPath: config.migrationsPath || "supabase/migrations",
            functionsPath: config.functionsPath || "supabase/functions",
            frontendPath: config.frontendPath || "frontend/src",
            expectedRpcFunctions: config.expectedRpcFunctions || [
                "get_videos_feed",
                "get_user_likes",
                "get_user_liked_tags",
                "get_user_liked_performers"
            ],
            ...config
        };

        this.rpcFunctions = [];
        this.rpcUsage = [];
        this.issues = [];
    }

    /**
     * スキーマ整合性チェック実行
     */
    async checkSchemaConsistency() {
        console.log(chalk.blue("🔍 データベーススキーマ整合性チェック開始..."));

        try {
            // RPC関数定義を解析
            await this.analyzeRpcDefinitions();

            // Edge Functions内のRPC使用を解析
            await this.analyzeRpcUsageInEdgeFunctions();

            // フロントエンド内のRPC使用を解析
            await this.analyzeRpcUsageInFrontend();

            // 整合性チェック実行
            this.performConsistencyChecks();

            console.log(chalk.green(`✅ チェック完了: ${this.issues.length}件の問題を発見`));

            return this.getResults();
        } catch (error) {
            console.error(chalk.red("❌ チェックエラー:"), error);
            throw error;
        }
    }

    /**
     * マイグレーションファイルからRPC関数定義を解析
     */
    async analyzeRpcDefinitions() {
        console.log(chalk.yellow("📄 RPC関数定義解析中..."));

        const migrationsPath = this.config.migrationsPath;
        if (!await fs.pathExists(migrationsPath)) {
            this.addIssue("missing_migrations", `マイグレーションディレクトリが見つかりません: ${migrationsPath}`);
            return;
        }

        const sqlFiles = await this.findSqlFiles(migrationsPath);

        for (const sqlFile of sqlFiles) {
            await this.analyzeSqlFile(sqlFile);
        }

        console.log(chalk.green(`📊 ${this.rpcFunctions.length}個のRPC関数を発見`));
    }

    /**
     * SQLファイルを解析してRPC関数定義を抽出
     */
    async analyzeSqlFile(sqlFile) {
        try {
            const content = await fs.readFile(sqlFile, 'utf8');

            // CREATE OR REPLACE FUNCTION パターンを検索
            const functionRegex = /CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s+RETURNS\s+(.*?)(?:\s+LANGUAGE\s+(\w+))?\s+(?:SECURITY\s+(\w+)\s+)?AS\s+\$([^$]*)\$([\s\S]*?)\$\6[^$]*;/gim;

            let match;
            while ((match = functionRegex.exec(content)) !== null) {
                const rpcFunction = {
                    name: match[1],
                    parameters: this.parseFunctionParameters(match[2]),
                    returnType: match[3].trim(),
                    language: match[4] || 'sql',
                    security: match[5] || 'invoker',
                    body: match[7],
                    file: sqlFile,
                    location: this.getLineNumber(content, match.index)
                };

                this.rpcFunctions.push(rpcFunction);
            }

            // DROP FUNCTION パターンも検索
            const dropRegex = /DROP\s+FUNCTION\s+(?:IF\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)/gim;
            while ((match = dropRegex.exec(content)) !== null) {
                // 削除された関数として記録
                this.addIssue("dropped_function", `関数 ${match[1]} が削除されています`, {
                    file: sqlFile,
                    line: this.getLineNumber(content, match.index)
                });
            }

        } catch (error) {
            this.addIssue("sql_parse_error", `SQLファイル解析エラー: ${sqlFile}`, error.message);
        }
    }

    /**
     * Edge Functions内のRPC使用を解析
     */
    async analyzeRpcUsageInEdgeFunctions() {
        console.log(chalk.yellow("⚡ Edge Functions内RPC使用解析中..."));

        const functionsPath = this.config.functionsPath;
        if (!await fs.pathExists(functionsPath)) {
            this.addIssue("missing_functions", `Edge Functionsディレクトリが見つかりません: ${functionsPath}`);
            return;
        }

        const task_queue = [];
        const task = parseFiles([functionsPath], (err, tree) => {
            if (err) return;

            const filename = tree.filename();

            // supabase.rpc() 呼び出しを検出
            tree.root().findAll({
                rule: {
                    pattern: "supabase.rpc($functionName, $params)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                const rpcUsage = {
                    functionName: this.extractStringValue(match.getMatch("functionName")),
                    parameters: match.getMatch("params")?.text(),
                    file: filename,
                    location: this.getAstLocation(match),
                    context: "edge_function"
                };
                this.rpcUsage.push(rpcUsage);
            });

            // sql`SELECT * FROM function_name()` パターンも検出
            tree.root().findAll({
                rule: {
                    pattern: "sql`$query`",
                    kind: "tagged_template_expression"
                }
            }).forEach(match => {
                const query = match.getMatch("query")?.text();
                if (query) {
                    const functionCalls = this.extractFunctionCallsFromSql(query);
                    functionCalls.forEach(funcName => {
                        this.rpcUsage.push({
                            functionName: funcName,
                            parameters: "unknown",
                            file: filename,
                            location: this.getAstLocation(match),
                            context: "sql_query"
                        });
                    });
                }
            });
        });

        task_queue.push(task);
        await Promise.all(task_queue);
    }

    /**
     * フロントエンド内のRPC使用を解析
     */
    async analyzeRpcUsageInFrontend() {
        console.log(chalk.yellow("🎨 フロントエンド内RPC使用解析中..."));

        const frontendPath = this.config.frontendPath;
        if (!await fs.pathExists(frontendPath)) {
            this.addIssue("missing_frontend", `フロントエンドディレクトリが見つかりません: ${frontendPath}`);
            return;
        }

        const task_queue = [];
        const task = parseFiles([frontendPath], (err, tree) => {
            if (err) return;

            const filename = tree.filename();

            // supabase.rpc() 呼び出しを検出
            tree.root().findAll({
                rule: {
                    pattern: "supabase.rpc($functionName, $params)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                const rpcUsage = {
                    functionName: this.extractStringValue(match.getMatch("functionName")),
                    parameters: match.getMatch("params")?.text(),
                    file: filename,
                    location: this.getAstLocation(match),
                    context: "frontend"
                };
                this.rpcUsage.push(rpcUsage);
            });
        });

        task_queue.push(task);
        await Promise.all(task_queue);
    }

    /**
     * 整合性チェック実行
     */
    performConsistencyChecks() {
        console.log(chalk.yellow("🔍 整合性チェック実行中..."));

        // RPC関数の存在チェック
        this.checkRpcFunctionExistence();

        // パラメータ整合性チェック
        this.checkParameterConsistency();

        // 期待される関数の存在チェック
        this.checkExpectedFunctions();

        // 未使用関数の検出
        this.checkUnusedFunctions();
    }

    /**
     * RPC関数の存在チェック
     */
    checkRpcFunctionExistence() {
        const definedFunctions = this.rpcFunctions.map(f => f.name);
        const usedFunctions = [...new Set(this.rpcUsage.map(u => u.functionName))];

        usedFunctions.forEach(funcName => {
            if (!definedFunctions.includes(funcName)) {
                const usages = this.rpcUsage.filter(u => u.functionName === funcName);
                usages.forEach(usage => {
                    this.addIssue(
                        "undefined_rpc_function",
                        `未定義のRPC関数「${funcName}」が使用されています`,
                        {
                            file: usage.file,
                            location: usage.location,
                            context: usage.context
                        }
                    );
                });
            }
        });
    }

    /**
     * パラメータ整合性チェック
     */
    checkParameterConsistency() {
        this.rpcUsage.forEach(usage => {
            const definition = this.rpcFunctions.find(f => f.name === usage.functionName);
            if (!definition) return;

            // パラメータ数や型の整合性をチェック（簡易版）
            if (usage.parameters && usage.parameters !== "unknown") {
                try {
                    const usedParams = this.parseUsageParameters(usage.parameters);
                    const definedParams = definition.parameters;

                    if (usedParams.length !== definedParams.length) {
                        this.addIssue(
                            "parameter_count_mismatch",
                            `RPC関数「${usage.functionName}」のパラメータ数が不一致です（定義: ${definedParams.length}, 使用: ${usedParams.length}）`,
                            {
                                file: usage.file,
                                location: usage.location,
                                expectedParams: definedParams,
                                actualParams: usedParams
                            }
                        );
                    }
                } catch (error) {
                    // パラメータ解析エラーは警告レベル
                    this.addIssue(
                        "parameter_parse_error",
                        `RPC関数「${usage.functionName}」のパラメータ解析に失敗しました`,
                        { file: usage.file, error: error.message }
                    );
                }
            }
        });
    }

    /**
     * 期待される関数の存在チェック
     */
    checkExpectedFunctions() {
        const definedFunctions = this.rpcFunctions.map(f => f.name);

        this.config.expectedRpcFunctions.forEach(expectedFunc => {
            if (!definedFunctions.includes(expectedFunc)) {
                this.addIssue(
                    "missing_expected_function",
                    `期待されるRPC関数「${expectedFunc}」が定義されていません`
                );
            }
        });
    }

    /**
     * 未使用関数の検出
     */
    checkUnusedFunctions() {
        const usedFunctions = [...new Set(this.rpcUsage.map(u => u.functionName))];

        this.rpcFunctions.forEach(definition => {
            if (!usedFunctions.includes(definition.name)) {
                this.addIssue(
                    "unused_rpc_function",
                    `未使用のRPC関数「${definition.name}」が定義されています`,
                    {
                        file: definition.file,
                        location: { line: definition.location }
                    }
                );
            }
        });
    }

    /**
     * ヘルパーメソッド
     */
    async findSqlFiles(directory) {
        const files = [];
        const entries = await fs.readdir(directory, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(directory, entry.name);
            if (entry.isDirectory()) {
                files.push(...await this.findSqlFiles(fullPath));
            } else if (entry.name.endsWith('.sql')) {
                files.push(fullPath);
            }
        }

        return files;
    }

    parseFunctionParameters(paramString) {
        if (!paramString.trim()) return [];

        return paramString.split(',').map(param => {
            const parts = param.trim().split(/\s+/);
            return {
                name: parts[0],
                type: parts.slice(1).join(' '),
                direction: parts.includes('IN') ? 'IN' : parts.includes('OUT') ? 'OUT' : 'IN'
            };
        });
    }

    parseUsageParameters(paramString) {
        try {
            // 簡易的なJSONパース（完全ではない）
            if (paramString.trim().startsWith('{') && paramString.trim().endsWith('}')) {
                const parsed = JSON.parse(paramString);
                return Object.keys(parsed);
            }
            return [];
        } catch {
            return [];
        }
    }

    extractFunctionCallsFromSql(query) {
        const functionRegex = /(\w+)\s*\(/g;
        const functions = [];
        let match;

        while ((match = functionRegex.exec(query)) !== null) {
            const funcName = match[1];
            // SQLキーワードを除外
            if (!this.isSqlKeyword(funcName)) {
                functions.push(funcName);
            }
        }

        return functions;
    }

    isSqlKeyword(word) {
        const keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX', 'VIEW',
            'FUNCTION', 'PROCEDURE', 'TRIGGER', 'DATABASE', 'SCHEMA'
        ];
        return keywords.includes(word.toUpperCase());
    }

    extractStringValue(node) {
        if (!node) return null;
        const text = node.text();
        return text.replace(/^['"`]|['"`]$/g, '');
    }

    getAstLocation(node) {
        const range = node.range();
        return {
            line: range.start.line + 1,
            column: range.start.column + 1,
            endLine: range.end.line + 1,
            endColumn: range.end.column + 1
        };
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

    getResults() {
        return {
            rpcFunctions: this.rpcFunctions,
            rpcUsage: this.rpcUsage,
            issues: this.issues,
            summary: {
                definedFunctions: this.rpcFunctions.length,
                usageInstances: this.rpcUsage.length,
                uniqueUsedFunctions: [...new Set(this.rpcUsage.map(u => u.functionName))].length,
                issuesFound: this.issues.length,
                timestamp: new Date().toISOString()
            }
        };
    }

    /**
     * 結果をJSONファイルに出力
     */
    async saveResults(outputPath = "reports/database-schema-check.json") {
        const results = this.getResults();

        const dir = path.dirname(outputPath);
        await fs.ensureDir(dir);

        await fs.writeFile(outputPath, JSON.stringify(results, null, 2));
        console.log(chalk.green(`📊 チェック結果を保存しました: ${outputPath}`));

        return outputPath;
    }

    /**
     * チェック結果のサマリーを表示
     */
    printSummary() {
        const results = this.getResults();

        console.log(chalk.blue("\n📋 データベーススキーマ整合性チェックサマリー"));
        console.log(`定義済みRPC関数: ${results.summary.definedFunctions}個`);
        console.log(`使用箇所: ${results.summary.usageInstances}箇所`);
        console.log(`使用されている関数: ${results.summary.uniqueUsedFunctions}個`);
        console.log(`問題発見: ${results.summary.issuesFound}件`);

        if (results.issues.length > 0) {
            console.log(chalk.red("\n⚠️  発見された問題:"));
            const issuesByType = {};
            results.issues.forEach(issue => {
                if (!issuesByType[issue.type]) {
                    issuesByType[issue.type] = [];
                }
                issuesByType[issue.type].push(issue);
            });

            Object.entries(issuesByType).forEach(([type, issues]) => {
                console.log(chalk.yellow(`\n${type} (${issues.length}件):`));
                issues.forEach((issue, index) => {
                    console.log(`  ${index + 1}. ${issue.message}`);
                });
            });
        }

        console.log(chalk.blue("\n📊 RPC関数使用状況:"));
        const usage = {};
        this.rpcUsage.forEach(u => {
            if (!usage[u.functionName]) {
                usage[u.functionName] = 0;
            }
            usage[u.functionName]++;
        });

        Object.entries(usage).forEach(([funcName, count]) => {
            console.log(`  ${funcName}: ${count}回使用`);
        });
    }
}

/**
 * CLI用の実行関数
 */
export async function checkDatabaseSchema(configPath = null) {
    let config = {};

    if (configPath && await fs.pathExists(configPath)) {
        const configContent = await fs.readFile(configPath, 'utf8');
        config = JSON.parse(configContent);
    }

    const checker = new DatabaseSchemaChecker(config);
    const results = await checker.checkSchemaConsistency();

    checker.printSummary();
    await checker.saveResults();

    return results;
}
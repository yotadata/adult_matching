import { parseFiles } from "@ast-grep/napi";
import { chalk, fs, path } from "zx";
import { errors } from "./errors.js";

/**
 * @typedef {import("@ast-grep/napi").SgNode} SgNode
 */

/**
 * Supabase整合性解析ツール
 * 既存のast_grep.jsを拡張してSupabase専用の解析機能を提供
 * 汎用的な設計で他のSupabaseプロジェクトでも再利用可能
 */
export class SupabaseAnalyzer {
    constructor(config = {}) {
        this.config = {
            frontendPath: config.frontendPath || "frontend/src",
            backendPath: config.backendPath || "supabase/functions",
            databasePath: config.databasePath || "supabase/migrations",
            outputPath: config.outputPath || "reports",
            skipPatterns: config.skipPatterns || [".test.", ".spec.", "node_modules"],
            includePatterns: config.includePatterns || [".ts", ".tsx", ".js", ".jsx", ".sql"],
            ...config
        };
        this.issues = [];
        this.results = {
            edgeFunctions: [],
            apiCalls: [],
            rpcFunctions: [],
            typeDefinitions: [],
            authPatterns: []
        };
    }

    /**
     * Supabase環境全体を解析
     */
    async analyzeSupabaseEnvironment() {
        console.log(chalk.blue("🔍 Supabase環境解析を開始..."));

        try {
            // Edge Functions解析
            await this.analyzeEdgeFunctions();

            // フロントエンドのAPI呼び出し解析
            await this.analyzeFrontendApiCalls();

            // データベースRPC関数解析
            await this.analyzeDatabaseRpcFunctions();

            // 型定義解析
            await this.analyzeTypeDefinitions();

            // 認証パターン解析
            await this.analyzeAuthenticationPatterns();

            console.log(chalk.green(`✅ 解析完了: ${this.issues.length}件の問題を発見`));

            return {
                results: this.results,
                issues: this.issues,
                summary: this.generateSummary()
            };
        } catch (error) {
            console.error(chalk.red("❌ 解析エラー:"), error);
            throw error;
        }
    }

    /**
     * Edge Functions解析
     */
    async analyzeEdgeFunctions() {
        const functionsPath = this.config.backendPath;
        if (!await fs.pathExists(functionsPath)) {
            this.addIssue("missing_directory", `Edge Functions ディレクトリが見つかりません: ${functionsPath}`);
            return;
        }

        console.log(chalk.yellow("📁 Edge Functions解析中..."));

        const task_queue = [];
        const task = parseFiles([functionsPath], (err, tree) => {
            if (err) {
                this.addIssue("parse_error", `ファイル解析エラー: ${tree.filename()}`, err);
                return;
            }

            const filename = tree.filename();
            const functionName = path.basename(path.dirname(filename));

            // Edge Function基本情報を抽出
            const functionInfo = {
                name: functionName,
                path: filename,
                exports: [],
                imports: [],
                httpHandlers: [],
                responseFormats: [],
                errorHandling: []
            };

            // HTTPハンドラー検出
            tree.root().findAll({
                rule: {
                    pattern: "serve($handler)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                functionInfo.httpHandlers.push({
                    handler: match.getMatch("handler")?.text(),
                    location: this.getLocation(match)
                });
            });

            // レスポンス形式検出
            tree.root().findAll({
                rule: {
                    pattern: "new Response($body, $options)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                functionInfo.responseFormats.push({
                    body: match.getMatch("body")?.text(),
                    options: match.getMatch("options")?.text(),
                    location: this.getLocation(match)
                });
            });

            // エラーハンドリング検出
            tree.root().findAll({
                rule: {
                    any: [
                        { pattern: "try { $body } catch ($error) { $handler }" },
                        { pattern: "throw new Error($message)" },
                        { pattern: "return new Response($error, { status: $code })" }
                    ]
                }
            }).forEach(match => {
                functionInfo.errorHandling.push({
                    type: this.detectErrorType(match),
                    location: this.getLocation(match)
                });
            });

            this.results.edgeFunctions.push(functionInfo);
        });

        task_queue.push(task);
        await Promise.all(task_queue);
    }

    /**
     * フロントエンドAPI呼び出し解析
     */
    async analyzeFrontendApiCalls() {
        const frontendPath = this.config.frontendPath;
        if (!await fs.pathExists(frontendPath)) {
            this.addIssue("missing_directory", `フロントエンドディレクトリが見つかりません: ${frontendPath}`);
            return;
        }

        console.log(chalk.yellow("🎨 フロントエンドAPI呼び出し解析中..."));

        const task_queue = [];
        const task = parseFiles([frontendPath], (err, tree) => {
            if (err) return;

            const filename = tree.filename();

            // supabase.functions.invoke 呼び出し検出
            tree.root().findAll({
                rule: {
                    pattern: "supabase.functions.invoke($functionName, $options)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                const apiCall = {
                    functionName: this.extractStringValue(match.getMatch("functionName")),
                    options: match.getMatch("options")?.text(),
                    file: filename,
                    location: this.getLocation(match)
                };
                this.results.apiCalls.push(apiCall);
            });

            // supabase.from() 呼び出し検出
            tree.root().findAll({
                rule: {
                    pattern: "supabase.from($table)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                const tableCall = {
                    table: this.extractStringValue(match.getMatch("table")),
                    file: filename,
                    location: this.getLocation(match),
                    type: "table_access"
                };
                this.results.apiCalls.push(tableCall);
            });

            // supabase.rpc() 呼び出し検出
            tree.root().findAll({
                rule: {
                    pattern: "supabase.rpc($functionName, $params)",
                    kind: "call_expression"
                }
            }).forEach(match => {
                const rpcCall = {
                    functionName: this.extractStringValue(match.getMatch("functionName")),
                    params: match.getMatch("params")?.text(),
                    file: filename,
                    location: this.getLocation(match),
                    type: "rpc_call"
                };
                this.results.apiCalls.push(rpcCall);
            });
        });

        task_queue.push(task);
        await Promise.all(task_queue);
    }

    /**
     * データベースRPC関数解析
     */
    async analyzeDatabaseRpcFunctions() {
        const dbPath = this.config.databasePath;
        if (!await fs.pathExists(dbPath)) {
            this.addIssue("missing_directory", `データベースディレクトリが見つかりません: ${dbPath}`);
            return;
        }

        console.log(chalk.yellow("🗄️ データベースRPC関数解析中..."));

        // SQLファイルからRPC関数定義を抽出
        const sqlFiles = await this.findFiles(dbPath, /\.sql$/);

        for (const sqlFile of sqlFiles) {
            const content = await fs.readFile(sqlFile, 'utf8');

            // CREATE OR REPLACE FUNCTION パターンを検索
            const functionMatches = content.matchAll(/CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(\w+)\s*\((.*?)\)\s+RETURNS\s+(.*?)\s+(?:LANGUAGE\s+\w+\s+)?AS\s+\$\$?([\s\S]*?)\$\$?/gi);

            for (const match of functionMatches) {
                const rpcFunction = {
                    name: match[1],
                    parameters: this.parseSqlParameters(match[2]),
                    returnType: match[3].trim(),
                    body: match[4],
                    file: sqlFile,
                    location: { line: this.getLineNumber(content, match.index) }
                };
                this.results.rpcFunctions.push(rpcFunction);
            }
        }
    }

    /**
     * 型定義解析
     */
    async analyzeTypeDefinitions() {
        console.log(chalk.yellow("📝 型定義解析中..."));

        const frontendPath = this.config.frontendPath;
        const task_queue = [];
        const task = parseFiles([frontendPath], (err, tree) => {
            if (err) return;

            const filename = tree.filename();

            // interface定義検出
            tree.root().findAll({
                rule: {
                    pattern: "interface $name { $body }",
                    kind: "interface_declaration"
                }
            }).forEach(match => {
                const typeDefinition = {
                    name: match.getMatch("name")?.text(),
                    body: match.getMatch("body")?.text(),
                    file: filename,
                    location: this.getLocation(match),
                    type: "interface"
                };
                this.results.typeDefinitions.push(typeDefinition);
            });

            // type定義検出
            tree.root().findAll({
                rule: {
                    pattern: "type $name = $definition",
                    kind: "type_alias_declaration"
                }
            }).forEach(match => {
                const typeDefinition = {
                    name: match.getMatch("name")?.text(),
                    definition: match.getMatch("definition")?.text(),
                    file: filename,
                    location: this.getLocation(match),
                    type: "type_alias"
                };
                this.results.typeDefinitions.push(typeDefinition);
            });
        });

        task_queue.push(task);
        await Promise.all(task_queue);
    }

    /**
     * 認証パターン解析
     */
    async analyzeAuthenticationPatterns() {
        console.log(chalk.yellow("🔐 認証パターン解析中..."));

        const paths = [this.config.frontendPath, this.config.backendPath];

        for (const basePath of paths) {
            if (!await fs.pathExists(basePath)) continue;

            const task_queue = [];
            const task = parseFiles([basePath], (err, tree) => {
                if (err) return;

                const filename = tree.filename();

                // 認証ヘッダー設定検出
                tree.root().findAll({
                    rule: {
                        any: [
                            { pattern: "Authorization: `Bearer ${$token}`" },
                            { pattern: "headers.Authorization = $value" },
                            { pattern: "getSession()" },
                            { pattern: "getUser()" }
                        ]
                    }
                }).forEach(match => {
                    const authPattern = {
                        pattern: match.text(),
                        file: filename,
                        location: this.getLocation(match),
                        category: this.categorizeAuthPattern(match.text())
                    };
                    this.results.authPatterns.push(authPattern);
                });
            });

            task_queue.push(task);
            await Promise.all(task_queue);
        }
    }

    /**
     * ヘルパーメソッド
     */
    addIssue(type, message, details = null) {
        this.issues.push({
            type,
            message,
            details,
            timestamp: new Date().toISOString()
        });
    }

    getLocation(node) {
        const range = node.range();
        return {
            line: range.start.line + 1,
            column: range.start.column + 1,
            endLine: range.end.line + 1,
            endColumn: range.end.column + 1
        };
    }

    extractStringValue(node) {
        if (!node) return null;
        const text = node.text();
        return text.replace(/^['"`]|['"`]$/g, '');
    }

    detectErrorType(match) {
        const text = match.text();
        if (text.includes('try')) return 'try_catch';
        if (text.includes('throw')) return 'throw_error';
        if (text.includes('Response')) return 'response_error';
        return 'unknown';
    }

    categorizeAuthPattern(text) {
        if (text.includes('Bearer')) return 'bearer_token';
        if (text.includes('Authorization')) return 'auth_header';
        if (text.includes('getSession')) return 'session_check';
        if (text.includes('getUser')) return 'user_check';
        return 'other';
    }

    parseSqlParameters(paramString) {
        if (!paramString.trim()) return [];
        return paramString.split(',').map(param => {
            const [name, type] = param.trim().split(/\s+/);
            return { name, type };
        });
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length;
    }

    async findFiles(dir, pattern) {
        const files = [];
        const entries = await fs.readdir(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                files.push(...await this.findFiles(fullPath, pattern));
            } else if (pattern.test(entry.name)) {
                files.push(fullPath);
            }
        }

        return files;
    }

    generateSummary() {
        return {
            edgeFunctionsCount: this.results.edgeFunctions.length,
            apiCallsCount: this.results.apiCalls.length,
            rpcFunctionsCount: this.results.rpcFunctions.length,
            typeDefinitionsCount: this.results.typeDefinitions.length,
            authPatternsCount: this.results.authPatterns.length,
            issuesCount: this.issues.length,
            analysisDate: new Date().toISOString()
        };
    }

    /**
     * 整合性チェック実行
     */
    checkConsistency() {
        console.log(chalk.blue("🔍 整合性チェック実行中..."));

        // Edge FunctionsとAPI呼び出しの整合性チェック
        this.checkEdgeFunctionConsistency();

        // RPC関数と呼び出しの整合性チェック
        this.checkRpcConsistency();

        // 型定義の整合性チェック
        this.checkTypeConsistency();

        // 認証パターンの整合性チェック
        this.checkAuthConsistency();
    }

    checkEdgeFunctionConsistency() {
        const functionNames = this.results.edgeFunctions.map(f => f.name);
        const apiCalls = this.results.apiCalls.filter(call => call.type !== 'table_access' && call.type !== 'rpc_call');

        apiCalls.forEach(call => {
            if (!functionNames.includes(call.functionName)) {
                this.addIssue(
                    "missing_edge_function",
                    `API呼び出し「${call.functionName}」に対応するEdge Functionが見つかりません`,
                    { file: call.file, location: call.location }
                );
            }
        });
    }

    checkRpcConsistency() {
        const rpcNames = this.results.rpcFunctions.map(f => f.name);
        const rpcCalls = this.results.apiCalls.filter(call => call.type === 'rpc_call');

        rpcCalls.forEach(call => {
            if (!rpcNames.includes(call.functionName)) {
                this.addIssue(
                    "missing_rpc_function",
                    `RPC呼び出し「${call.functionName}」に対応する関数定義が見つかりません`,
                    { file: call.file, location: call.location }
                );
            }
        });
    }

    checkTypeConsistency() {
        // VideoFromApi等の重要な型定義の存在確認
        const typeNames = this.results.typeDefinitions.map(t => t.name);
        const requiredTypes = ['VideoFromApi', 'CardData'];

        requiredTypes.forEach(typeName => {
            if (!typeNames.includes(typeName)) {
                this.addIssue(
                    "missing_type_definition",
                    `必要な型定義「${typeName}」が見つかりません`
                );
            }
        });
    }

    checkAuthConsistency() {
        const authHeaderUsage = this.results.authPatterns.filter(p => p.category === 'auth_header');
        const sessionChecks = this.results.authPatterns.filter(p => p.category === 'session_check');

        if (authHeaderUsage.length === 0 && sessionChecks.length > 0) {
            this.addIssue(
                "auth_inconsistency",
                "セッションチェックはありますが、認証ヘッダーの設定が見つかりません"
            );
        }
    }
}

/**
 * CLI用の実行関数
 */
export async function analyzeSupabaseProject(configPath = null) {
    let config = {};

    if (configPath && await fs.pathExists(configPath)) {
        const configContent = await fs.readFile(configPath, 'utf8');
        config = JSON.parse(configContent);
    }

    const analyzer = new SupabaseAnalyzer(config);

    const analysisResult = await analyzer.analyzeSupabaseEnvironment();
    analyzer.checkConsistency();

    return {
        ...analysisResult,
        issues: analyzer.issues
    };
}
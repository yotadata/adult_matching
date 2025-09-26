import { chalk, fs, path } from "zx";

/**
 * Edge Functions レスポンス形式バリデーター
 * 汎用的なSupabase Edge Function検証ツール
 * 他のSupabaseプロジェクトでも再利用可能
 */
export class EdgeFunctionValidator {
    constructor(config = {}) {
        this.config = {
            supabaseUrl: config.supabaseUrl || process.env.SUPABASE_URL,
            supabaseKey: config.supabaseKey || process.env.SUPABASE_ANON_KEY,
            functionsPath: config.functionsPath || "supabase/functions",
            testTimeout: config.testTimeout || 10000,
            expectedFunctions: config.expectedFunctions || [
                "videos-feed",
                "likes",
                "update_user_embedding",
                "delete_account"
            ],
            ...config
        };

        this.validationResults = [];
        this.issues = [];
    }

    /**
     * 全Edge Functionsの検証を実行
     */
    async validateAllFunctions() {
        console.log(chalk.blue("🔍 Edge Functions レスポンス形式検証開始..."));

        if (!this.config.supabaseUrl || !this.config.supabaseKey) {
            this.addIssue("missing_config", "Supabase URL または API Key が設定されていません");
            return this.getResults();
        }

        // 存在するEdge Functionsを検出
        const availableFunctions = await this.discoverFunctions();

        // 各関数の検証実行
        for (const functionName of this.config.expectedFunctions) {
            if (availableFunctions.includes(functionName)) {
                await this.validateFunction(functionName);
            } else {
                this.addIssue("missing_function", `Edge Function「${functionName}」が見つかりません`);
            }
        }

        // レスポンス形式の統一性をチェック
        this.checkResponseConsistency();

        console.log(chalk.green(`✅ 検証完了: ${this.issues.length}件の問題を発見`));
        return this.getResults();
    }

    /**
     * 利用可能なEdge Functionsを検出
     */
    async discoverFunctions() {
        const functionsPath = this.config.functionsPath;
        const functions = [];

        try {
            if (await fs.pathExists(functionsPath)) {
                const entries = await fs.readdir(functionsPath, { withFileTypes: true });
                for (const entry of entries) {
                    if (entry.isDirectory()) {
                        const indexPath = path.join(functionsPath, entry.name, "index.ts");
                        if (await fs.pathExists(indexPath)) {
                            functions.push(entry.name);
                        }
                    }
                }
            }
        } catch (error) {
            this.addIssue("discovery_error", `Functions検出エラー: ${error.message}`);
        }

        console.log(chalk.yellow(`📁 検出されたFunctions: ${functions.join(", ")}`));
        return functions;
    }

    /**
     * 特定のEdge Functionを検証
     */
    async validateFunction(functionName) {
        console.log(chalk.yellow(`🧪 ${functionName} 検証中...`));

        const validation = {
            functionName,
            timestamp: new Date().toISOString(),
            tests: {
                httpMethods: [],
                responseFormat: null,
                errorHandling: null,
                authHandling: null
            },
            issues: []
        };

        try {
            // 基本的なHTTPメソッドテスト
            await this.testHttpMethods(functionName, validation);

            // レスポンス形式テスト
            await this.testResponseFormat(functionName, validation);

            // エラーハンドリングテスト
            await this.testErrorHandling(functionName, validation);

            // 認証ハンドリングテスト
            await this.testAuthHandling(functionName, validation);

        } catch (error) {
            validation.issues.push({
                type: "validation_error",
                message: `検証エラー: ${error.message}`
            });
        }

        this.validationResults.push(validation);
    }

    /**
     * HTTPメソッドテスト
     */
    async testHttpMethods(functionName, validation) {
        const methods = ["GET", "POST", "DELETE"];

        for (const method of methods) {
            try {
                const response = await this.callFunction(functionName, {
                    method,
                    body: method === "GET" ? undefined : JSON.stringify({ test: true })
                });

                validation.tests.httpMethods.push({
                    method,
                    status: response.status,
                    success: response.ok || response.status === 405, // 405は許可される
                    contentType: response.headers.get("content-type")
                });

            } catch (error) {
                validation.tests.httpMethods.push({
                    method,
                    success: false,
                    error: error.message
                });
            }
        }
    }

    /**
     * レスポンス形式テスト
     */
    async testResponseFormat(functionName, validation) {
        try {
            const response = await this.callFunction(functionName, {
                method: this.getPreferredMethod(functionName)
            });

            const contentType = response.headers.get("content-type");
            let responseBody;

            try {
                responseBody = await response.json();
            } catch {
                responseBody = await response.text();
            }

            validation.tests.responseFormat = {
                status: response.status,
                contentType,
                isJson: contentType?.includes("application/json"),
                hasErrorField: this.hasStandardErrorField(responseBody),
                hasDataField: this.hasStandardDataField(responseBody),
                structure: this.analyzeResponseStructure(responseBody)
            };

            // レスポンス形式の問題を検出
            if (!validation.tests.responseFormat.isJson) {
                validation.issues.push({
                    type: "response_format",
                    message: "JSONレスポンスが期待されますが、異なる形式です"
                });
            }

        } catch (error) {
            validation.tests.responseFormat = {
                error: error.message,
                success: false
            };
        }
    }

    /**
     * エラーハンドリングテスト
     */
    async testErrorHandling(functionName, validation) {
        try {
            // 不正なリクエストで意図的にエラーを発生させる
            const response = await this.callFunction(functionName, {
                method: "POST",
                body: JSON.stringify({ invalid: "data".repeat(10000) }) // 大きなデータ
            });

            let errorResponse;
            try {
                errorResponse = await response.json();
            } catch {
                errorResponse = await response.text();
            }

            validation.tests.errorHandling = {
                status: response.status,
                hasErrorMessage: this.hasErrorMessage(errorResponse),
                errorStructure: this.analyzeErrorStructure(errorResponse),
                isConsistent: this.isConsistentErrorFormat(errorResponse)
            };

        } catch (error) {
            validation.tests.errorHandling = {
                error: error.message,
                success: false
            };
        }
    }

    /**
     * 認証ハンドリングテスト
     */
    async testAuthHandling(functionName, validation) {
        try {
            // 認証なしでのアクセステスト
            const responseNoAuth = await this.callFunction(functionName, {
                method: this.getPreferredMethod(functionName),
                headers: {} // 認証ヘッダーなし
            });

            // 不正な認証でのアクセステスト
            const responseInvalidAuth = await this.callFunction(functionName, {
                method: this.getPreferredMethod(functionName),
                headers: {
                    "Authorization": "Bearer invalid-token"
                }
            });

            validation.tests.authHandling = {
                noAuthStatus: responseNoAuth.status,
                invalidAuthStatus: responseInvalidAuth.status,
                handlesAuthProperly: responseNoAuth.status === 401 || responseInvalidAuth.status === 401,
                allowsUnauthenticated: responseNoAuth.status === 200
            };

        } catch (error) {
            validation.tests.authHandling = {
                error: error.message,
                success: false
            };
        }
    }

    /**
     * レスポンス形式の統一性チェック
     */
    checkResponseConsistency() {
        console.log(chalk.yellow("📊 レスポンス形式統一性チェック中..."));

        const successResponses = this.validationResults
            .filter(v => v.tests.responseFormat && v.tests.responseFormat.isJson)
            .map(v => v.tests.responseFormat);

        if (successResponses.length === 0) return;

        // 共通のレスポンス構造パターンを分析
        const structures = successResponses.map(r => r.structure);
        const commonFields = this.findCommonFields(structures);

        // 不統一な形式を検出
        successResponses.forEach((response, index) => {
            const validation = this.validationResults[index];

            if (!this.hasRequiredFields(response.structure, ["status", "data"])) {
                this.addIssue(
                    "inconsistent_response",
                    `${validation.functionName}: 標準的なレスポンス形式（status, data）に準拠していません`
                );
            }
        });

        // エラーレスポンスの統一性チェック
        const errorResponses = this.validationResults
            .filter(v => v.tests.errorHandling && v.tests.errorHandling.errorStructure)
            .map(v => v.tests.errorHandling);

        errorResponses.forEach((errorTest, index) => {
            const validation = this.validationResults[index];

            if (!errorTest.isConsistent) {
                this.addIssue(
                    "inconsistent_error",
                    `${validation.functionName}: エラーレスポンス形式が統一されていません`
                );
            }
        });
    }

    /**
     * ヘルパーメソッド
     */
    async callFunction(functionName, options = {}) {
        const url = `${this.config.supabaseUrl}/functions/v1/${functionName}`;

        const defaultHeaders = {
            "apikey": this.config.supabaseKey,
            "Content-Type": "application/json"
        };

        const fetchOptions = {
            method: options.method || "GET",
            headers: { ...defaultHeaders, ...options.headers },
            ...options
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.testTimeout);

        try {
            const response = await fetch(url, {
                ...fetchOptions,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    getPreferredMethod(functionName) {
        const methodMap = {
            "videos-feed": "POST",
            "likes": "GET",
            "update_user_embedding": "POST",
            "delete_account": "POST"
        };
        return methodMap[functionName] || "GET";
    }

    hasStandardErrorField(response) {
        if (typeof response !== "object") return false;
        return "error" in response || "message" in response;
    }

    hasStandardDataField(response) {
        if (typeof response !== "object") return false;
        return "data" in response || Array.isArray(response);
    }

    analyzeResponseStructure(response) {
        if (typeof response !== "object") {
            return { type: "non-object", value: typeof response };
        }

        return {
            type: "object",
            fields: Object.keys(response),
            fieldCount: Object.keys(response).length,
            hasNesting: Object.values(response).some(v => typeof v === "object")
        };
    }

    hasErrorMessage(response) {
        if (typeof response !== "object") return false;
        return "error" in response || "message" in response || "detail" in response;
    }

    analyzeErrorStructure(response) {
        if (typeof response !== "object") {
            return { type: "non-object" };
        }

        return {
            hasError: "error" in response,
            hasMessage: "message" in response,
            hasCode: "code" in response || "status" in response,
            fields: Object.keys(response)
        };
    }

    isConsistentErrorFormat(response) {
        if (typeof response !== "object") return false;

        // 標準的なエラー形式パターンをチェック
        const standardPatterns = [
            ["error", "message"],
            ["error", "details"],
            ["message", "code"],
            ["error"]
        ];

        const responseFields = Object.keys(response);
        return standardPatterns.some(pattern =>
            pattern.every(field => responseFields.includes(field))
        );
    }

    findCommonFields(structures) {
        if (structures.length === 0) return [];

        return structures[0].fields.filter(field =>
            structures.every(structure => structure.fields.includes(field))
        );
    }

    hasRequiredFields(structure, requiredFields) {
        return requiredFields.every(field => structure.fields.includes(field));
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
            validationResults: this.validationResults,
            issues: this.issues,
            summary: {
                functionsValidated: this.validationResults.length,
                issuesFound: this.issues.length,
                successfulValidations: this.validationResults.filter(v => v.issues.length === 0).length,
                timestamp: new Date().toISOString()
            }
        };
    }

    /**
     * 結果をJSONファイルに出力
     */
    async saveResults(outputPath = "reports/edge-function-validation.json") {
        const results = this.getResults();

        // ディレクトリが存在しない場合は作成
        const dir = path.dirname(outputPath);
        await fs.ensureDir(dir);

        await fs.writeFile(outputPath, JSON.stringify(results, null, 2));
        console.log(chalk.green(`📊 検証結果を保存しました: ${outputPath}`));

        return outputPath;
    }

    /**
     * 検証結果のサマリーを表示
     */
    printSummary() {
        const results = this.getResults();

        console.log(chalk.blue("\n📋 Edge Functions 検証サマリー"));
        console.log(`検証対象: ${results.summary.functionsValidated}個のFunction`);
        console.log(`成功: ${results.summary.successfulValidations}個`);
        console.log(`問題発見: ${results.summary.issuesFound}件`);

        if (results.issues.length > 0) {
            console.log(chalk.red("\n⚠️  発見された問題:"));
            results.issues.forEach((issue, index) => {
                console.log(chalk.red(`${index + 1}. [${issue.type}] ${issue.message}`));
            });
        }

        console.log(chalk.blue("\n📊 Function別結果:"));
        results.validationResults.forEach(validation => {
            const status = validation.issues.length === 0 ?
                chalk.green("✅ OK") :
                chalk.red(`❌ ${validation.issues.length}件の問題`);
            console.log(`  ${validation.functionName}: ${status}`);
        });
    }
}

/**
 * CLI用の実行関数
 */
export async function validateEdgeFunctions(configPath = null) {
    let config = {};

    if (configPath && await fs.pathExists(configPath)) {
        const configContent = await fs.readFile(configPath, 'utf8');
        config = JSON.parse(configContent);
    }

    const validator = new EdgeFunctionValidator(config);
    const results = await validator.validateAllFunctions();

    validator.printSummary();
    await validator.saveResults();

    return results;
}
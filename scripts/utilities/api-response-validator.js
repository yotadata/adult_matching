import { chalk, fs, path } from "zx";

/**
 * API レスポンス実環境バリデーター
 * 汎用API検証ツール
 * 実際のAPIエンドポイントを呼び出してレスポンス形式を検証
 */
export class ApiResponseValidator {
    constructor(config = {}) {
        this.config = {
            supabaseUrl: config.supabaseUrl || process.env.SUPABASE_URL,
            supabaseKey: config.supabaseKey || process.env.SUPABASE_ANON_KEY,
            testTimeout: config.testTimeout || 15000,
            maxRetries: config.maxRetries || 3,
            retryDelay: config.retryDelay || 1000,
            testData: config.testData || {},
            environments: config.environments || ['development'],
            ...config
        };

        this.validationResults = [];
        this.environmentResults = {};
        this.issues = [];
    }

    /**
     * 実環境API検証実行
     */
    async validateApiResponses() {
        console.log(chalk.blue("🔍 API レスポンス実環境検証開始..."));

        if (!this.config.supabaseUrl || !this.config.supabaseKey) {
            this.addIssue("missing_config", "Supabase設定が不完全です");
            return this.getResults();
        }

        // 各環境での検証実行
        for (const env of this.config.environments) {
            console.log(chalk.cyan(`🌍 環境「${env}」での検証開始...`));
            this.environmentResults[env] = await this.validateEnvironment(env);
        }

        // 環境間整合性チェック
        if (this.config.environments.length > 1) {
            this.checkEnvironmentConsistency();
        }

        console.log(chalk.green(`✅ 検証完了: ${this.issues.length}件の問題を発見`));
        return this.getResults();
    }

    /**
     * 特定環境での検証
     */
    async validateEnvironment(environment) {
        const results = {
            environment,
            timestamp: new Date().toISOString(),
            endpoints: {},
            summary: {
                tested: 0,
                passed: 0,
                failed: 0
            }
        };

        // Edge Functions検証
        const edgeFunctions = [
            { name: "videos-feed", method: "POST", auth: true },
            { name: "likes", method: "GET", auth: true },
            { name: "update_user_embedding", method: "POST", auth: true },
            { name: "delete_account", method: "POST", auth: true }
        ];

        for (const func of edgeFunctions) {
            console.log(chalk.yellow(`  📡 ${func.name} 検証中...`));
            results.endpoints[func.name] = await this.validateEndpoint(func, environment);
            results.summary.tested++;

            if (results.endpoints[func.name].success) {
                results.summary.passed++;
            } else {
                results.summary.failed++;
            }
        }

        return results;
    }

    /**
     * 個別エンドポイント検証
     */
    async validateEndpoint(endpoint, environment) {
        const validation = {
            endpoint: endpoint.name,
            method: endpoint.method,
            environment,
            timestamp: new Date().toISOString(),
            tests: {
                connectivity: null,
                authentication: null,
                responseFormat: null,
                errorHandling: null,
                performance: null
            },
            success: false,
            issues: []
        };

        try {
            // 接続性テスト
            validation.tests.connectivity = await this.testConnectivity(endpoint);

            // 認証テスト
            if (endpoint.auth) {
                validation.tests.authentication = await this.testAuthentication(endpoint);
            }

            // レスポンス形式テスト
            validation.tests.responseFormat = await this.testResponseFormat(endpoint);

            // エラーハンドリングテスト
            validation.tests.errorHandling = await this.testErrorHandling(endpoint);

            // パフォーマンステスト
            validation.tests.performance = await this.testPerformance(endpoint);

            // 総合判定
            validation.success = this.evaluateSuccess(validation.tests);

        } catch (error) {
            validation.issues.push({
                type: "validation_error",
                message: `検証エラー: ${error.message}`
            });
        }

        return validation;
    }

    /**
     * 接続性テスト
     */
    async testConnectivity(endpoint) {
        const test = {
            name: "connectivity",
            success: false,
            responseTime: null,
            status: null,
            error: null
        };

        const startTime = Date.now();

        try {
            const response = await this.makeRequest(endpoint.name, {
                method: endpoint.method,
                headers: this.getAuthHeaders()
            });

            test.responseTime = Date.now() - startTime;
            test.status = response.status;
            test.success = response.status < 500; // 5xxエラー以外は接続性OK

        } catch (error) {
            test.error = error.message;
            test.responseTime = Date.now() - startTime;
        }

        return test;
    }

    /**
     * 認証テスト
     */
    async testAuthentication(endpoint) {
        const test = {
            name: "authentication",
            scenarios: {
                noAuth: null,
                invalidAuth: null,
                validAuth: null
            },
            success: false
        };

        try {
            // 認証なし
            const noAuthResponse = await this.makeRequest(endpoint.name, {
                method: endpoint.method,
                headers: {}
            });
            test.scenarios.noAuth = {
                status: noAuthResponse.status,
                expectsAuth: noAuthResponse.status === 401
            };

            // 不正な認証
            const invalidAuthResponse = await this.makeRequest(endpoint.name, {
                method: endpoint.method,
                headers: { "Authorization": "Bearer invalid-token" }
            });
            test.scenarios.invalidAuth = {
                status: invalidAuthResponse.status,
                rejectsInvalid: invalidAuthResponse.status === 401
            };

            // 有効な認証（テスト用）
            const validAuthResponse = await this.makeRequest(endpoint.name, {
                method: endpoint.method,
                headers: this.getAuthHeaders()
            });
            test.scenarios.validAuth = {
                status: validAuthResponse.status,
                acceptsValid: validAuthResponse.status < 400
            };

            test.success = test.scenarios.validAuth.acceptsValid;

        } catch (error) {
            test.error = error.message;
        }

        return test;
    }

    /**
     * レスポンス形式テスト
     */
    async testResponseFormat(endpoint) {
        const test = {
            name: "responseFormat",
            success: false,
            contentType: null,
            isJson: false,
            structure: null,
            schema: null,
            error: null
        };

        try {
            const response = await this.makeRequest(endpoint.name, {
                method: endpoint.method,
                headers: this.getAuthHeaders(),
                body: this.getTestData(endpoint.name)
            });

            test.contentType = response.headers.get("content-type");
            test.isJson = test.contentType?.includes("application/json");

            if (test.isJson) {
                const data = await response.json();
                test.structure = this.analyzeStructure(data);
                test.schema = this.generateSchema(data);
                test.success = this.validateResponseSchema(data, endpoint.name);
            } else {
                const text = await response.text();
                test.structure = { type: "text", length: text.length };
            }

        } catch (error) {
            test.error = error.message;
        }

        return test;
    }

    /**
     * エラーハンドリングテスト
     */
    async testErrorHandling(endpoint) {
        const test = {
            name: "errorHandling",
            scenarios: [],
            success: false
        };

        const errorScenarios = [
            {
                name: "invalid_data",
                data: { invalid: "x".repeat(100000) }
            },
            {
                name: "malformed_json",
                data: "{ invalid json",
                contentType: "application/json"
            },
            {
                name: "wrong_method",
                method: endpoint.method === "GET" ? "POST" : "GET"
            }
        ];

        for (const scenario of errorScenarios) {
            try {
                const response = await this.makeRequest(endpoint.name, {
                    method: scenario.method || endpoint.method,
                    headers: {
                        ...this.getAuthHeaders(),
                        "Content-Type": scenario.contentType || "application/json"
                    },
                    body: scenario.data ? JSON.stringify(scenario.data) : undefined
                });

                const isError = response.status >= 400;
                let errorData = null;

                if (isError) {
                    try {
                        errorData = await response.json();
                    } catch {
                        errorData = await response.text();
                    }
                }

                test.scenarios.push({
                    name: scenario.name,
                    status: response.status,
                    hasError: isError,
                    errorData,
                    hasErrorMessage: this.hasErrorMessage(errorData),
                    isStructured: typeof errorData === "object"
                });

            } catch (error) {
                test.scenarios.push({
                    name: scenario.name,
                    error: error.message
                });
            }
        }

        test.success = test.scenarios.every(s => s.hasError && s.hasErrorMessage);
        return test;
    }

    /**
     * パフォーマンステスト
     */
    async testPerformance(endpoint) {
        const test = {
            name: "performance",
            measurements: [],
            average: 0,
            min: 0,
            max: 0,
            success: false
        };

        const iterations = 3;

        for (let i = 0; i < iterations; i++) {
            const startTime = Date.now();

            try {
                await this.makeRequest(endpoint.name, {
                    method: endpoint.method,
                    headers: this.getAuthHeaders(),
                    body: this.getTestData(endpoint.name)
                });

                const responseTime = Date.now() - startTime;
                test.measurements.push(responseTime);

            } catch (error) {
                test.measurements.push(null);
            }

            // リクエスト間隔
            if (i < iterations - 1) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }

        const validMeasurements = test.measurements.filter(m => m !== null);

        if (validMeasurements.length > 0) {
            test.average = validMeasurements.reduce((a, b) => a + b, 0) / validMeasurements.length;
            test.min = Math.min(...validMeasurements);
            test.max = Math.max(...validMeasurements);
            test.success = test.average < 5000; // 5秒以内
        }

        return test;
    }

    /**
     * 環境間整合性チェック
     */
    checkEnvironmentConsistency() {
        console.log(chalk.yellow("🔄 環境間整合性チェック中..."));

        const environments = Object.keys(this.environmentResults);
        if (environments.length < 2) return;

        const baseEnv = environments[0];
        const baseResults = this.environmentResults[baseEnv];

        for (let i = 1; i < environments.length; i++) {
            const compareEnv = environments[i];
            const compareResults = this.environmentResults[compareEnv];

            // エンドポイント比較
            Object.keys(baseResults.endpoints).forEach(endpoint => {
                const baseEndpoint = baseResults.endpoints[endpoint];
                const compareEndpoint = compareResults.endpoints[endpoint];

                if (!compareEndpoint) {
                    this.addIssue(
                        "environment_inconsistency",
                        `エンドポイント「${endpoint}」が環境「${compareEnv}」に存在しません`
                    );
                    return;
                }

                // レスポンス形式の一致チェック
                if (baseEndpoint.tests.responseFormat?.schema &&
                    compareEndpoint.tests.responseFormat?.schema) {
                    const schemaMatch = this.compareSchemas(
                        baseEndpoint.tests.responseFormat.schema,
                        compareEndpoint.tests.responseFormat.schema
                    );

                    if (!schemaMatch) {
                        this.addIssue(
                            "schema_inconsistency",
                            `エンドポイント「${endpoint}」のレスポンススキーマが環境間で異なります`,
                            {
                                baseEnv,
                                compareEnv,
                                endpoint
                            }
                        );
                    }
                }
            });
        }
    }

    /**
     * ヘルパーメソッド
     */
    async makeRequest(functionName, options = {}) {
        const url = `${this.config.supabaseUrl}/functions/v1/${functionName}`;

        const fetchOptions = {
            method: options.method || "GET",
            headers: options.headers || {},
            body: options.body,
            signal: AbortSignal.timeout(this.config.testTimeout)
        };

        let lastError;
        for (let i = 0; i < this.config.maxRetries; i++) {
            try {
                const response = await fetch(url, fetchOptions);
                return response;
            } catch (error) {
                lastError = error;
                if (i < this.config.maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
                }
            }
        }

        throw lastError;
    }

    getAuthHeaders() {
        return {
            "apikey": this.config.supabaseKey,
            "Authorization": `Bearer ${this.config.supabaseKey}`,
            "Content-Type": "application/json"
        };
    }

    getTestData(endpoint) {
        const testData = {
            "videos-feed": JSON.stringify({}),
            "likes": undefined,
            "update_user_embedding": JSON.stringify({
                batch_phase: "test",
                batch_size: 10,
                completed_items: 5
            }),
            "delete_account": JSON.stringify({})
        };

        return testData[endpoint] || undefined;
    }

    analyzeStructure(data) {
        if (Array.isArray(data)) {
            return {
                type: "array",
                length: data.length,
                itemType: data.length > 0 ? typeof data[0] : "unknown"
            };
        } else if (typeof data === "object" && data !== null) {
            return {
                type: "object",
                fields: Object.keys(data),
                fieldCount: Object.keys(data).length
            };
        } else {
            return {
                type: typeof data,
                value: data
            };
        }
    }

    generateSchema(data) {
        if (Array.isArray(data)) {
            return {
                type: "array",
                items: data.length > 0 ? this.generateSchema(data[0]) : {}
            };
        } else if (typeof data === "object" && data !== null) {
            const schema = { type: "object", properties: {} };
            Object.keys(data).forEach(key => {
                schema.properties[key] = this.generateSchema(data[key]);
            });
            return schema;
        } else {
            return { type: typeof data };
        }
    }

    validateResponseSchema(data, endpoint) {
        // 基本的なスキーマ検証
        const expectedSchemas = {
            "videos-feed": { type: "array" },
            "likes": { type: "object", required: ["likes"] },
            "update_user_embedding": { type: "object" },
            "delete_account": { type: "object" }
        };

        const expected = expectedSchemas[endpoint];
        if (!expected) return true;

        if (expected.type === "array" && !Array.isArray(data)) return false;
        if (expected.type === "object" && typeof data !== "object") return false;

        if (expected.required) {
            return expected.required.every(field => field in data);
        }

        return true;
    }

    hasErrorMessage(errorData) {
        if (typeof errorData !== "object") return false;
        return "error" in errorData || "message" in errorData || "detail" in errorData;
    }

    compareSchemas(schema1, schema2) {
        return JSON.stringify(schema1) === JSON.stringify(schema2);
    }

    evaluateSuccess(tests) {
        const criticalTests = ["connectivity", "responseFormat"];
        return criticalTests.every(test => tests[test]?.success !== false);
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
            environmentResults: this.environmentResults,
            issues: this.issues,
            summary: {
                environmentsTested: Object.keys(this.environmentResults).length,
                totalIssues: this.issues.length,
                timestamp: new Date().toISOString()
            }
        };
    }

    /**
     * 結果出力
     */
    async saveResults(outputPath = "reports/api-response-validation.json") {
        const results = this.getResults();

        const dir = path.dirname(outputPath);
        await fs.ensureDir(dir);

        await fs.writeFile(outputPath, JSON.stringify(results, null, 2));
        console.log(chalk.green(`📊 検証結果を保存しました: ${outputPath}`));

        return outputPath;
    }

    printSummary() {
        const results = this.getResults();

        console.log(chalk.blue("\n📋 API レスポンス検証サマリー"));
        console.log(`検証環境: ${results.summary.environmentsTested}環境`);
        console.log(`問題発見: ${results.summary.totalIssues}件`);

        Object.entries(results.environmentResults).forEach(([env, envResult]) => {
            console.log(chalk.cyan(`\n🌍 環境: ${env}`));
            console.log(`  テスト実行: ${envResult.summary.tested}件`);
            console.log(`  成功: ${envResult.summary.passed}件`);
            console.log(`  失敗: ${envResult.summary.failed}件`);
        });

        if (results.issues.length > 0) {
            console.log(chalk.red("\n⚠️  発見された問題:"));
            results.issues.forEach((issue, index) => {
                console.log(chalk.red(`${index + 1}. [${issue.type}] ${issue.message}`));
            });
        }
    }
}

/**
 * CLI用の実行関数
 */
export async function validateApiResponses(configPath = null) {
    let config = {};

    if (configPath && await fs.pathExists(configPath)) {
        const configContent = await fs.readFile(configPath, 'utf8');
        config = JSON.parse(configContent);
    }

    const validator = new ApiResponseValidator(config);
    const results = await validator.validateApiResponses();

    validator.printSummary();
    await validator.saveResults();

    return results;
}
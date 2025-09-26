/**
 * 統一入力検証・バリデーションユーティリティ
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * 基本的な入力検証
 */
export class Validator {
  private errors: string[] = [];

  /**
   * 必須フィールドをチェック
   */
  required(value: any, fieldName: string): this {
    if (value === null || value === undefined || value === '') {
      this.errors.push(`${fieldName} is required`);
    }
    return this;
  }

  /**
   * 文字列の長さをチェック
   */
  stringLength(value: string, min: number, max: number, fieldName: string): this {
    if (typeof value === 'string') {
      if (value.length < min || value.length > max) {
        this.errors.push(`${fieldName} must be between ${min} and ${max} characters`);
      }
    }
    return this;
  }

  /**
   * 数値の範囲をチェック
   */
  numberRange(value: number, min: number, max: number, fieldName: string): this {
    if (typeof value === 'number') {
      if (value < min || value > max) {
        this.errors.push(`${fieldName} must be between ${min} and ${max}`);
      }
    }
    return this;
  }

  /**
   * 配列の長さをチェック
   */
  arrayLength(value: any[], min: number, max: number, fieldName: string): this {
    if (Array.isArray(value)) {
      if (value.length < min || value.length > max) {
        this.errors.push(`${fieldName} must have between ${min} and ${max} items`);
      }
    }
    return this;
  }

  /**
   * UUID形式をチェック
   */
  uuid(value: string, fieldName: string): this {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    if (typeof value === 'string' && !uuidRegex.test(value)) {
      this.errors.push(`${fieldName} must be a valid UUID`);
    }
    return this;
  }

  /**
   * メールアドレス形式をチェック
   */
  email(value: string, fieldName: string): this {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (typeof value === 'string' && !emailRegex.test(value)) {
      this.errors.push(`${fieldName} must be a valid email address`);
    }
    return this;
  }

  /**
   * 検証結果を取得
   */
  getResult(): ValidationResult {
    return {
      valid: this.errors.length === 0,
      errors: [...this.errors]
    };
  }

  /**
   * エラーをリセット
   */
  reset(): this {
    this.errors = [];
    return this;
  }
}

/**
 * リクエストペイロードの共通検証
 */
export interface PaginationValidation {
  limit?: number;
  offset?: number;
  page?: number;
}

export function validatePagination(params: PaginationValidation): ValidationResult {
  const validator = new Validator();

  if (params.limit !== undefined) {
    validator.numberRange(params.limit, 1, 100, 'limit');
  }

  if (params.offset !== undefined) {
    validator.numberRange(params.offset, 0, Number.MAX_SAFE_INTEGER, 'offset');
  }

  if (params.page !== undefined) {
    validator.numberRange(params.page, 1, Number.MAX_SAFE_INTEGER, 'page');
  }

  return validator.getResult();
}

/**
 * フィード関連パラメータの検証
 */
export interface FeedValidation {
  feed_type?: string;
  limit?: number;
  offset?: number;
  exclude_ids?: string[];
  user_id?: string;
}

export function validateFeedRequest(params: FeedValidation): ValidationResult {
  const validator = new Validator();

  if (params.feed_type) {
    const validTypes = ['explore', 'personalized', 'latest', 'popular', 'random'];
    if (!validTypes.includes(params.feed_type)) {
      validator.errors.push(`feed_type must be one of: ${validTypes.join(', ')}`);
    }
  }

  if (params.user_id) {
    validator.uuid(params.user_id, 'user_id');
  }

  if (params.exclude_ids && Array.isArray(params.exclude_ids)) {
    params.exclude_ids.forEach((id, index) => {
      validator.uuid(id, `exclude_ids[${index}]`);
    });
  }

  // ページネーション検証も含める
  const paginationResult = validatePagination(params);
  validator.errors.push(...paginationResult.errors);

  return validator.getResult();
}

/**
 * 推薦関連パラメータの検証
 */
export interface RecommendationValidation {
  user_id: string;
  limit?: number;
  offset?: number;
  exclude_ids?: string[];
  model_version?: string;
}

export function validateRecommendationRequest(params: RecommendationValidation): ValidationResult {
  const validator = new Validator();

  validator
    .required(params.user_id, 'user_id')
    .uuid(params.user_id, 'user_id');

  if (params.model_version) {
    validator.stringLength(params.model_version, 1, 50, 'model_version');
  }

  if (params.exclude_ids && Array.isArray(params.exclude_ids)) {
    params.exclude_ids.forEach((id, index) => {
      validator.uuid(id, `exclude_ids[${index}]`);
    });
  }

  // ページネーション検証
  const paginationResult = validatePagination(params);
  validator.errors.push(...paginationResult.errors);

  return validator.getResult();
}

/**
 * 共通JSON解析と検証
 */
export async function parseAndValidate<T>(
  req: Request,
  validator: (data: any) => ValidationResult
): Promise<{ data?: T; errors?: string[] }> {
  try {
    const data = await req.json();
    const validation = validator(data);
    
    if (!validation.valid) {
      return { errors: validation.errors };
    }

    return { data };
  } catch (error) {
    return { errors: ['Invalid JSON payload'] };
  }
}
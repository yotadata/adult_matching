/**
 * エラー収集システム
 * 解析中に発生したエラーを収集・管理するためのユーティリティ
 */

export const errors = [];

export function clearErrors() {
    errors.length = 0;
}

export function addError(error) {
    errors.push(error);
}

export function getErrors() {
    return [...errors];
}

export function hasErrors() {
    return errors.length > 0;
}

export function getErrorCount() {
    return errors.length;
}
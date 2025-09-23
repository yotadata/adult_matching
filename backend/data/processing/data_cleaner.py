"""
Data Cleaner

データクリーニングとデータ品質向上ユーティリティ
不正データの検出・修正・標準化を実行
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from urllib.parse import urlparse

from backend.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CleaningRule:
    """データクリーニングルール"""
    name: str
    description: str
    column: str
    rule_type: str  # "standardize", "validate", "transform", "remove", "fix"
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = None
    active: bool = True

@dataclass
class CleaningResult:
    """クリーニング結果"""
    rule_name: str
    records_processed: int = 0
    records_modified: int = 0
    records_removed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DataCleaner:
    """データクリーニングシステム"""
    
    def __init__(self):
        self.cleaning_rules = self._get_default_rules()
        self.results: List[CleaningResult] = []
        
    def _get_default_rules(self) -> List[CleaningRule]:
        """デフォルトクリーニングルール"""
        return [
            # テキスト標準化
            CleaningRule(
                name="title_standardization",
                description="Standardize title format",
                column="title",
                rule_type="standardize",
                function=self._standardize_title
            ),
            CleaningRule(
                name="description_cleanup",
                description="Clean description text",
                column="description", 
                rule_type="standardize",
                function=self._clean_description
            ),
            
            # URL検証・修正
            CleaningRule(
                name="url_validation",
                description="Validate and fix URLs",
                column="thumbnail_url",
                rule_type="validate",
                function=self._validate_url
            ),
            CleaningRule(
                name="sample_url_validation",
                description="Validate sample video URLs",
                column="sample_video_url",
                rule_type="validate",
                function=self._validate_url
            ),
            
            # 価格データ標準化
            CleaningRule(
                name="price_standardization",
                description="Standardize price format",
                column="price",
                rule_type="standardize",
                function=self._standardize_price
            ),
            
            # 日付標準化
            CleaningRule(
                name="date_standardization",
                description="Standardize date formats",
                column="release_date",
                rule_type="standardize",
                function=self._standardize_date
            ),
            
            # 配列データクリーニング
            CleaningRule(
                name="performers_cleanup",
                description="Clean performers array",
                column="performers",
                rule_type="standardize",
                function=self._clean_array_field
            ),
            CleaningRule(
                name="tags_cleanup",
                description="Clean tags array",
                column="tags",
                rule_type="standardize",
                function=self._clean_array_field
            ),
            CleaningRule(
                name="image_urls_cleanup",
                description="Clean image URLs array",
                column="image_urls",
                rule_type="standardize",
                function=self._clean_url_array
            ),
            
            # 重複除去
            CleaningRule(
                name="duplicate_removal",
                description="Remove duplicate records",
                column="external_id",
                rule_type="remove",
                function=self._remove_duplicates
            )
        ]
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        custom_rules: Optional[List[CleaningRule]] = None,
        skip_rules: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[CleaningResult]]:
        """
        データフレームのクリーニング実行
        
        Args:
            df: クリーニング対象データフレーム
            custom_rules: カスタムクリーニングルール
            skip_rules: スキップするルール名リスト
            
        Returns:
            (クリーニング済みデータフレーム, クリーニング結果リスト)
        """
        logger.info(f"🧹 Starting data cleaning for {len(df)} records")
        
        cleaned_df = df.copy()
        self.results = []
        
        # 使用するルール決定
        rules_to_apply = custom_rules or self.cleaning_rules
        active_rules = [r for r in rules_to_apply if r.active]
        
        if skip_rules:
            active_rules = [r for r in active_rules if r.name not in skip_rules]
        
        # ルール適用
        for rule in active_rules:
            try:
                logger.debug(f"Applying rule: {rule.name}")
                cleaned_df, result = self._apply_cleaning_rule(cleaned_df, rule)
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {e}")
                error_result = CleaningResult(
                    rule_name=rule.name,
                    errors=[f"Rule application error: {str(e)}"]
                )
                self.results.append(error_result)
        
        logger.info(f"✅ Data cleaning completed. {len(cleaned_df)} records after cleaning")
        return cleaned_df, self.results
    
    def _apply_cleaning_rule(
        self,
        df: pd.DataFrame,
        rule: CleaningRule
    ) -> Tuple[pd.DataFrame, CleaningResult]:
        """クリーニングルール適用"""
        result = CleaningResult(rule_name=rule.name)
        
        if rule.column not in df.columns:
            result.errors.append(f"Column {rule.column} not found")
            return df, result
        
        original_count = len(df)
        result.records_processed = original_count
        
        try:
            if rule.function:
                if rule.rule_type == "remove":
                    # 削除系ルール
                    cleaned_df = rule.function(df)
                    result.records_removed = original_count - len(cleaned_df)
                else:
                    # 変換系ルール
                    cleaned_df = df.copy()
                    original_values = cleaned_df[rule.column].copy()
                    cleaned_df[rule.column] = rule.function(cleaned_df[rule.column])
                    
                    # 変更された記録数をカウント
                    if not original_values.equals(cleaned_df[rule.column]):
                        changed_mask = original_values != cleaned_df[rule.column]
                        result.records_modified = changed_mask.sum()
            else:
                result.errors.append("No function defined for rule")
                return df, result
                
        except Exception as e:
            result.errors.append(f"Function execution error: {str(e)}")
            return df, result
        
        return cleaned_df, result
    
    # データクリーニング関数群
    def _standardize_title(self, series: pd.Series) -> pd.Series:
        """タイトル標準化"""
        return series.astype(str).apply(lambda x: self._clean_text(x, max_length=200))
    
    def _clean_description(self, series: pd.Series) -> pd.Series:
        """説明文クリーニング"""
        return series.astype(str).apply(lambda x: self._clean_text(x, max_length=1000))
    
    def _clean_text(self, text: str, max_length: int = 500) -> str:
        """テキストクリーニング"""
        if pd.isna(text) or text == 'nan':
            return ""
        
        # 基本クリーニング
        text = str(text).strip()
        
        # 特殊文字・制御文字除去
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # 複数の空白を単一の空白に
        text = re.sub(r'\s+', ' ', text)
        
        # HTML タグ除去（基本的なもの）
        text = re.sub(r'<[^>]+>', '', text)
        
        # 長さ制限
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        
        return text.strip()
    
    def _validate_url(self, series: pd.Series) -> pd.Series:
        """URL検証・修正"""
        def fix_url(url):
            if pd.isna(url) or str(url).strip() == '' or str(url) == 'nan':
                return ""
            
            url_str = str(url).strip()
            
            # 基本的なURL形式チェック
            if not url_str.startswith(('http://', 'https://')):
                if url_str.startswith('//'):
                    url_str = 'https:' + url_str
                elif url_str.startswith('/'):
                    # 相対URLは修正不可
                    return ""
                else:
                    # プロトコルなしの場合はhttpsを追加
                    url_str = 'https://' + url_str
            
            # URL妥当性チェック
            try:
                parsed = urlparse(url_str)
                if parsed.netloc and parsed.scheme in ['http', 'https']:
                    return url_str
                else:
                    return ""
            except:
                return ""
        
        return series.apply(fix_url)
    
    def _standardize_price(self, series: pd.Series) -> pd.Series:
        """価格標準化"""
        def clean_price(price):
            if pd.isna(price):
                return 0
            
            if isinstance(price, (int, float)):
                return max(0, int(price))
            
            # 文字列の場合の処理
            price_str = str(price).strip()
            
            # 数字以外の文字を除去
            price_clean = re.sub(r'[^\d]', '', price_str)
            
            try:
                return max(0, int(price_clean)) if price_clean else 0
            except:
                return 0
        
        return series.apply(clean_price)
    
    def _standardize_date(self, series: pd.Series) -> pd.Series:
        """日付標準化"""
        def clean_date(date_val):
            if pd.isna(date_val) or str(date_val).strip() == '' or str(date_val) == 'nan':
                return None
            
            try:
                # 複数の日付形式に対応
                date_str = str(date_val).strip()
                
                # 既にISO形式の場合
                if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', date_str):
                    return date_str
                
                # YYYY-MM-DD HH:MM:SS 形式
                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date_str):
                    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    return dt.isoformat()
                
                # YYYY-MM-DD 形式
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    return dt.isoformat()
                
                return None
            except:
                return None
        
        return series.apply(clean_date)
    
    def _clean_array_field(self, series: pd.Series) -> pd.Series:
        """配列フィールドクリーニング"""
        def clean_array(arr):
            if pd.isna(arr) or arr == 'nan':
                return []
            
            if isinstance(arr, str):
                try:
                    # JSON文字列の場合
                    parsed = json.loads(arr)
                    if isinstance(parsed, list):
                        arr = parsed
                    else:
                        return []
                except:
                    # カンマ区切りの場合
                    arr = [item.strip() for item in arr.split(',')]
            
            if isinstance(arr, list):
                # 空白・重複除去
                cleaned = []
                for item in arr:
                    cleaned_item = str(item).strip()
                    if cleaned_item and cleaned_item not in cleaned:
                        cleaned.append(cleaned_item)
                return cleaned
            
            return []
        
        return series.apply(clean_array)
    
    def _clean_url_array(self, series: pd.Series) -> pd.Series:
        """URL配列クリーニング"""
        def clean_url_array(arr):
            cleaned_arr = self._clean_array_field(pd.Series([arr])).iloc[0]
            
            # 各URLを検証
            valid_urls = []
            for url in cleaned_arr:
                cleaned_url = self._validate_url(pd.Series([url])).iloc[0]
                if cleaned_url:
                    valid_urls.append(cleaned_url)
            
            return valid_urls
        
        return series.apply(clean_url_array)
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """重複除去"""
        # external_idとsourceの組み合わせで重複チェック
        if 'source' in df.columns:
            duplicated_mask = df.duplicated(subset=['external_id', 'source'], keep='first')
        else:
            duplicated_mask = df.duplicated(subset=['external_id'], keep='first')
        
        return df[~duplicated_mask].copy()
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """クリーニング結果サマリー"""
        if not self.results:
            return {"message": "No cleaning results available"}
        
        total_processed = sum(r.records_processed for r in self.results)
        total_modified = sum(r.records_modified for r in self.results)
        total_removed = sum(r.records_removed for r in self.results)
        
        rule_summaries = []
        for result in self.results:
            rule_summaries.append({
                "rule_name": result.rule_name,
                "processed": result.records_processed,
                "modified": result.records_modified,
                "removed": result.records_removed,
                "errors": result.errors
            })
        
        return {
            "summary": {
                "total_rules_applied": len(self.results),
                "total_records_processed": total_processed,
                "total_records_modified": total_modified,
                "total_records_removed": total_removed
            },
            "rule_details": rule_summaries
        }


# 便利関数
def clean_data(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """データクリーニング実行（簡単インターフェース）"""
    cleaner = DataCleaner()
    cleaned_df, results = cleaner.clean_dataframe(df, **kwargs)
    summary = cleaner.get_cleaning_summary()
    return cleaned_df, summary
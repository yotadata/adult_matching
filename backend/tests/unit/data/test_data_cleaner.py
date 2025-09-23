"""
Tests for DataCleaner

データクリーニングシステムの単体テスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import json

# テスト対象のモジュールをインポート
try:
    from backend.data.processing.data_cleaner import (
        DataCleaner, 
        CleaningRule, 
        CleaningResult,
        clean_data
    )
except ImportError:
    pytest.skip("Data processing modules not found", allow_module_level=True)


class TestCleaningRule:
    """CleaningRuleの単体テスト"""
    
    @pytest.mark.unit
    def test_cleaning_rule_creation(self):
        """CleaningRule作成テスト"""
        rule = CleaningRule(
            name="test_rule",
            description="Test rule",
            column="title",
            rule_type="standardize"
        )
        
        assert rule.name == "test_rule"
        assert rule.description == "Test rule"
        assert rule.column == "title"
        assert rule.rule_type == "standardize"
        assert rule.function is None
        assert rule.parameters == {}
        assert rule.active is True


class TestCleaningResult:
    """CleaningResultの単体テスト"""
    
    @pytest.mark.unit
    def test_cleaning_result_creation(self):
        """CleaningResult作成テスト"""
        result = CleaningResult(rule_name="test_rule")
        
        assert result.rule_name == "test_rule"
        assert result.records_processed == 0
        assert result.records_modified == 0
        assert result.records_removed == 0
        assert result.errors == []
    
    @pytest.mark.unit
    def test_cleaning_result_with_errors(self):
        """エラー付きCleaningResult作成テスト"""
        errors = ["Error 1", "Error 2"]
        result = CleaningResult(
            rule_name="test_rule",
            errors=errors
        )
        
        assert result.errors == errors


class TestDataCleaner:
    """DataCleanerの単体テスト"""
    
    @pytest.fixture
    def sample_dirty_data(self):
        """汚れたテストデータ"""
        return pd.DataFrame({
            'title': [
                '  Test Video Title  ',
                'Another Title<script>alert("xss")</script>',
                'Very Long Title ' * 20,
                '',
                None
            ],
            'description': [
                'Good description',
                'Description with\x00control\x01chars',
                'Normal description',
                '',
                'nan'
            ],
            'price': [
                '1,980円',
                1980,
                'invalid_price',
                '0',
                None
            ],
            'thumbnail_url': [
                'https://example.com/thumb.jpg',
                '//example.com/thumb.jpg',
                'not_a_url',
                '',
                None
            ],
            'performers': [
                '["actor1", "actor2"]',
                'actor1,actor2,actor3',
                '',
                None,
                '[]'
            ],
            'external_id': [
                'id1',
                'id2',
                'id3',
                'id1',  # 重複
                'id5'
            ],
            'source': [
                'dmm',
                'dmm',
                'dmm',
                'dmm',  # 重複
                'dmm'
            ]
        })
    
    @pytest.fixture
    def data_cleaner(self):
        """DataCleanerインスタンス"""
        return DataCleaner()
    
    @pytest.mark.unit
    def test_initialization(self, data_cleaner):
        """初期化テスト"""
        assert len(data_cleaner.cleaning_rules) > 0
        assert data_cleaner.results == []
    
    @pytest.mark.unit
    def test_default_rules_loaded(self, data_cleaner):
        """デフォルトルール読み込みテスト"""
        rule_names = [rule.name for rule in data_cleaner.cleaning_rules]
        
        expected_rules = [
            'title_standardization',
            'description_cleanup',
            'url_validation',
            'price_standardization',
            'duplicate_removal'
        ]
        
        for expected_rule in expected_rules:
            assert expected_rule in rule_names
    
    @pytest.mark.unit
    def test_title_standardization(self, data_cleaner):
        """タイトル標準化テスト"""
        test_series = pd.Series([
            '  Test Title  ',
            'Title\x00with\x01control',
            'Very Long Title ' * 50,
            '',
            'nan'
        ])
        
        result = data_cleaner._standardize_title(test_series)
        
        assert result.iloc[0] == 'Test Title'
        assert '\x00' not in result.iloc[1]
        assert len(result.iloc[2]) <= 200
        assert result.iloc[3] == ''
        assert result.iloc[4] == ''
    
    @pytest.mark.unit
    def test_price_standardization(self, data_cleaner):
        """価格標準化テスト"""
        test_series = pd.Series([
            '1,980円',
            1980,
            'invalid',
            '0',
            None,
            '2,500円'
        ])
        
        result = data_cleaner._standardize_price(test_series)
        
        assert result.iloc[0] == 1980
        assert result.iloc[1] == 1980
        assert result.iloc[2] == 0  # 無効な価格は0
        assert result.iloc[3] == 0
        assert result.iloc[4] == 0  # Noneは0
        assert result.iloc[5] == 2500
    
    @pytest.mark.unit
    def test_url_validation(self, data_cleaner):
        """URL検証テスト"""
        test_series = pd.Series([
            'https://example.com/valid.jpg',
            '//example.com/protocol-relative.jpg',
            'invalid_url',
            '',
            None,
            '/relative/path'
        ])
        
        result = data_cleaner._validate_url(test_series)
        
        assert result.iloc[0] == 'https://example.com/valid.jpg'
        assert result.iloc[1] == 'https://example.com/protocol-relative.jpg'
        assert result.iloc[2] == ''  # 無効なURL
        assert result.iloc[3] == ''
        assert result.iloc[4] == ''
        assert result.iloc[5] == ''  # 相対パス
    
    @pytest.mark.unit
    def test_array_field_cleaning(self, data_cleaner):
        """配列フィールドクリーニングテスト"""
        test_series = pd.Series([
            '["item1", "item2"]',
            'item1,item2,item3',
            '',
            None,
            'nan',
            '[]'
        ])
        
        result = data_cleaner._clean_array_field(test_series)
        
        assert result.iloc[0] == ['item1', 'item2']
        assert result.iloc[1] == ['item1', 'item2', 'item3']
        assert result.iloc[2] == []
        assert result.iloc[3] == []
        assert result.iloc[4] == []
        assert result.iloc[5] == []
    
    @pytest.mark.unit
    def test_duplicate_removal(self, data_cleaner, sample_dirty_data):
        """重複除去テスト"""
        result_df = data_cleaner._remove_duplicates(sample_dirty_data)
        
        # 重複が除去されていることを確認
        assert len(result_df) == 4  # 元5行から1行減る
        
        # external_id + sourceの組み合わせでユニークであることを確認
        unique_combinations = result_df.groupby(['external_id', 'source']).size()
        assert all(unique_combinations == 1)
    
    @pytest.mark.unit
    def test_clean_dataframe_basic(self, data_cleaner, sample_dirty_data):
        """基本的なデータフレームクリーニングテスト"""
        cleaned_df, results = data_cleaner.clean_dataframe(sample_dirty_data)
        
        # 結果の検証
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(results, list)
        assert len(results) > 0
        
        # 重複が除去されていることを確認
        assert len(cleaned_df) == 4
        
        # タイトルが標準化されていることを確認
        assert cleaned_df.iloc[0]['title'].strip() == cleaned_df.iloc[0]['title']
    
    @pytest.mark.unit
    def test_clean_dataframe_with_skip_rules(self, data_cleaner, sample_dirty_data):
        """ルールスキップ付きクリーニングテスト"""
        skip_rules = ['duplicate_removal']
        cleaned_df, results = data_cleaner.clean_dataframe(
            sample_dirty_data, 
            skip_rules=skip_rules
        )
        
        # 重複除去がスキップされているので元の行数のまま
        assert len(cleaned_df) == 5
        
        # スキップしたルールの結果が含まれていないことを確認
        rule_names = [r.rule_name for r in results]
        assert 'duplicate_removal' not in rule_names
    
    @pytest.mark.unit
    def test_custom_cleaning_rule(self, data_cleaner, sample_dirty_data):
        """カスタムクリーニングルールテスト"""
        def custom_function(series):
            return series.str.upper()
        
        custom_rule = CleaningRule(
            name="custom_uppercase",
            description="Convert to uppercase",
            column="title",
            rule_type="transform",
            function=custom_function
        )
        
        cleaned_df, results = data_cleaner.clean_dataframe(
            sample_dirty_data,
            custom_rules=[custom_rule]
        )
        
        # カスタムルールの結果が含まれていることを確認
        rule_names = [r.rule_name for r in results]
        assert 'custom_uppercase' in rule_names
    
    @pytest.mark.unit
    def test_cleaning_with_missing_column(self, data_cleaner):
        """存在しないカラムでのクリーニングテスト"""
        df = pd.DataFrame({'title': ['Test']})
        
        # 存在しないカラムのルールを作成
        custom_rule = CleaningRule(
            name="missing_column_rule",
            description="Rule for missing column",
            column="missing_column",
            rule_type="standardize",
            function=lambda x: x
        )
        
        cleaned_df, results = data_cleaner.clean_dataframe(
            df,
            custom_rules=[custom_rule]
        )
        
        # エラーが記録されていることを確認
        missing_column_result = next(
            r for r in results if r.rule_name == "missing_column_rule"
        )
        assert len(missing_column_result.errors) > 0
        assert "not found" in missing_column_result.errors[0]
    
    @pytest.mark.unit
    def test_get_cleaning_summary(self, data_cleaner, sample_dirty_data):
        """クリーニングサマリー取得テスト"""
        # クリーニング実行
        data_cleaner.clean_dataframe(sample_dirty_data)
        
        summary = data_cleaner.get_cleaning_summary()
        
        assert 'summary' in summary
        assert 'rule_details' in summary
        assert 'total_rules_applied' in summary['summary']
        assert 'total_records_processed' in summary['summary']
        assert isinstance(summary['rule_details'], list)
    
    @pytest.mark.unit
    def test_empty_dataframe_handling(self, data_cleaner):
        """空のデータフレーム処理テスト"""
        empty_df = pd.DataFrame()
        
        cleaned_df, results = data_cleaner.clean_dataframe(empty_df)
        
        assert len(cleaned_df) == 0
        assert isinstance(results, list)
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_large_dataset_performance(self, data_cleaner, performance_timer):
        """大量データセットのパフォーマンステスト"""
        # 大量データ作成
        large_data = pd.DataFrame({
            'title': [f'Title {i}' for i in range(10000)],
            'description': [f'Description {i}' for i in range(10000)],
            'price': [1000 + i for i in range(10000)],
            'external_id': [f'id_{i}' for i in range(10000)],
            'source': ['dmm'] * 10000
        })
        
        performance_timer.start()
        cleaned_df, results = data_cleaner.clean_dataframe(large_data)
        performance_timer.stop()
        
        assert len(cleaned_df) == 10000
        assert performance_timer.duration < 30.0  # 30秒以内


class TestCleanDataFunction:
    """clean_data関数の単体テスト"""
    
    @pytest.mark.unit
    def test_clean_data_function(self, sample_video_data):
        """clean_data便利関数テスト"""
        df = pd.DataFrame([sample_video_data])
        
        cleaned_df, summary = clean_data(df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert 'summary' in summary
        assert 'rule_details' in summary
"""
Data Pipeline Integration Tests

データ処理パイプライン統合テスト - データフロー全体のテスト
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tests.integration.data.mock_processors import UnifiedDataProcessor, DataQualityMonitor, PipelineManager


@pytest.fixture
def sample_raw_data():
    """サンプル生データ"""
    return {
        'videos': [
            {
                'external_id': 'raw001',
                'source': 'dmm',
                'title': '  テスト動画1  ',  # トリムが必要
                'description': 'これはテスト用の動画です。',
                'maker': 'テストメーカー',
                'genre': 'アクション,ドラマ',  # 分割が必要
                'price': '1000',  # 数値変換が必要
                'thumbnail_url': 'https://example.com/thumb1.jpg',
                'duration': '120分',  # 秒数変換が必要
                'release_date': '2024-01-15',
                'rating': '4.5',
                'review_count': '42'
            },
            {
                'external_id': 'raw002',
                'source': 'dmm',
                'title': 'テスト動画2',
                'description': '',  # 空の説明
                'maker': '',  # 空のメーカー
                'genre': 'コメディ',
                'price': '無料',  # 特殊価格処理が必要
                'thumbnail_url': 'https://example.com/thumb2.jpg',
                'duration': '90',  # 単位なし
                'release_date': '2024-01-16',
                'rating': None,  # NULL値処理
                'review_count': '0'
            },
            {
                'external_id': '',  # 無効データ
                'source': 'dmm',
                'title': 'テスト動画3',
                'description': 'テスト説明3',
                'maker': 'テストメーカー3',
                'genre': 'ドラマ',
                'price': '-100',  # 負の価格
                'thumbnail_url': 'invalid-url',  # 無効URL
                'duration': '不明',  # 無効時間
                'release_date': '無効日付',
                'rating': '6.0',  # 範囲外評価
                'review_count': '-5'  # 負のレビュー数
            }
        ],
        'users': [
            {
                'user_id': 'user001',
                'age': '25',
                'gender': 'M',
                'prefecture': '東京都',
                'occupation': 'エンジニア',
                'interests': 'アクション,コメディ',
                'signup_date': '2024-01-01'
            },
            {
                'user_id': 'user002',
                'age': 'unknown',  # 無効年齢
                'gender': 'F',
                'prefecture': '',
                'occupation': '学生',
                'interests': 'ドラマ',
                'signup_date': '2024-01-02'
            }
        ]
    }


@pytest.fixture
def temp_pipeline_dir():
    """一時パイプラインディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="pipeline_integration_test_")
    
    # 必要なサブディレクトリを作成
    subdirs = ['raw', 'processed', 'validated', 'exported', 'logs']
    for subdir in subdirs:
        (Path(temp_dir) / subdir).mkdir(exist_ok=True)
    
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def pipeline_config(temp_pipeline_dir):
    """パイプライン設定"""
    return {
        'data_dir': temp_pipeline_dir,
        'raw_data_dir': Path(temp_pipeline_dir) / 'raw',
        'processed_data_dir': Path(temp_pipeline_dir) / 'processed',
        'validated_data_dir': Path(temp_pipeline_dir) / 'validated',
        'exported_data_dir': Path(temp_pipeline_dir) / 'exported',
        'logs_dir': Path(temp_pipeline_dir) / 'logs',
        'batch_size': 100,
        'max_workers': 2,
        'quality_threshold': 0.8,
        'enable_validation': True,
        'enable_monitoring': True
    }


class TestDataPipelineIntegration:
    """データパイプライン統合テストクラス"""
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_ingestion_pipeline(self, sample_raw_data, pipeline_config):
        """データ取り込みパイプラインテスト"""
        pipeline_manager = PipelineManager(config=pipeline_config)
        
        # 生データをファイルに保存
        raw_data_file = pipeline_config['raw_data_dir'] / 'test_data.json'
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_raw_data, f, ensure_ascii=False, indent=2)
        
        # 取り込み処理実行
        ingestion_result = pipeline_manager.ingest_data(
            source_path=str(raw_data_file),
            data_type='mixed'
        )
        
        # 結果検証
        assert ingestion_result['success'] == True
        assert ingestion_result['total_records'] == len(sample_raw_data['videos']) + len(sample_raw_data['users'])
        assert ingestion_result['videos_count'] == len(sample_raw_data['videos'])
        assert ingestion_result['users_count'] == len(sample_raw_data['users'])
        
        # 処理されたデータファイルの存在確認
        processed_files = list(pipeline_config['processed_data_dir'].glob('*.json'))
        assert len(processed_files) > 0
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_cleaning_pipeline(self, sample_raw_data, pipeline_config):
        """データクリーニングパイプラインテスト"""
        processor = UnifiedDataProcessor(config=pipeline_config)
        
        # クリーニング処理実行
        cleaning_result = processor.clean_data(sample_raw_data)
        
        # 動画データクリーニング結果検証
        cleaned_videos = cleaning_result['videos']
        
        # 有効なデータの確認
        valid_videos = [v for v in cleaned_videos if v.get('is_valid', False)]
        assert len(valid_videos) >= 2  # 3つ中2つは有効であることを期待
        
        for video in valid_videos:
            # データ型変換の確認
            assert isinstance(video['price'], (int, float))
            assert isinstance(video['duration_seconds'], int)
            assert isinstance(video['rating'], (int, float, type(None)))
            
            # データクリーニングの確認
            assert video['title'].strip() == video['title']  # トリム済み
            assert video['external_id'] != ''  # 空でない
            
            # URL検証
            if video.get('thumbnail_url'):
                assert video['thumbnail_url'].startswith('http')
        
        # 無効データの除外確認
        invalid_videos = [v for v in cleaned_videos if not v.get('is_valid', True)]
        assert len(invalid_videos) >= 1  # 少なくとも1つは無効データ
        
        # ユーザーデータクリーニング結果検証
        cleaned_users = cleaning_result['users']
        valid_users = [u for u in cleaned_users if u.get('is_valid', False)]
        
        for user in valid_users:
            # 年齢の数値変換確認
            if user.get('age'):
                assert isinstance(user['age'], int)
                assert 0 <= user['age'] <= 120
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_validation_pipeline(self, sample_raw_data, pipeline_config):
        """データ検証パイプラインテスト"""
        validator = DataValidator()
        processor = UnifiedDataProcessor(config=pipeline_config)
        
        # まずクリーニング
        cleaned_data = processor.clean_data(sample_raw_data)
        
        # 検証実行
        validation_result = validator.validate_pipeline_data(cleaned_data)
        
        # 検証結果の確認
        assert 'videos' in validation_result
        assert 'users' in validation_result
        assert 'summary' in validation_result
        
        # サマリー情報の確認
        summary = validation_result['summary']
        assert 'total_records' in summary
        assert 'valid_records' in summary
        assert 'invalid_records' in summary
        assert 'validation_rate' in summary
        
        # 検証率の確認（品質要件）
        validation_rate = summary['validation_rate']
        assert 0.0 <= validation_rate <= 1.0
        
        # 個別レコード検証結果の確認
        video_validations = validation_result['videos']
        for validation in video_validations:
            assert 'record_id' in validation
            assert 'is_valid' in validation
            assert 'errors' in validation
            assert 'warnings' in validation
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_quality_monitoring(self, sample_raw_data, pipeline_config):
        """データ品質監視テスト"""
        quality_monitor = QualityMonitor()
        processor = UnifiedDataProcessor(config=pipeline_config)
        
        # データ処理
        processed_data = processor.clean_data(sample_raw_data)
        
        # 品質監視実行
        quality_report = quality_monitor.generate_quality_report(processed_data)
        
        # レポート構造の確認
        assert 'overall_score' in quality_report
        assert 'metrics' in quality_report
        assert 'recommendations' in quality_report
        assert 'timestamp' in quality_report
        
        # 品質スコアの確認
        overall_score = quality_report['overall_score']
        assert 0.0 <= overall_score <= 1.0
        
        # メトリクスの確認
        metrics = quality_report['metrics']
        required_metrics = [
            'completeness', 'accuracy', 'consistency', 
            'validity', 'uniqueness'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
        
        # 推奨事項の確認
        recommendations = quality_report['recommendations']
        assert isinstance(recommendations, list)
        
        # 品質閾値テスト
        if overall_score < pipeline_config['quality_threshold']:
            assert len(recommendations) > 0  # 品質が低い場合は推奨事項があること
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.slow
    def test_end_to_end_pipeline(self, sample_raw_data, pipeline_config):
        """エンドツーエンドパイプラインテスト"""
        pipeline_manager = PipelineManager(config=pipeline_config)
        
        # 生データファイル作成
        raw_data_file = pipeline_config['raw_data_dir'] / 'e2e_test_data.json'
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_raw_data, f, ensure_ascii=False, indent=2)
        
        # パイプライン全体実行
        start_time = datetime.now()
        
        pipeline_result = pipeline_manager.run_full_pipeline(
            source_path=str(raw_data_file),
            output_format='json',
            enable_quality_check=True
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 実行結果検証
        assert pipeline_result['success'] == True
        assert 'ingestion' in pipeline_result
        assert 'cleaning' in pipeline_result
        assert 'validation' in pipeline_result
        assert 'quality_check' in pipeline_result
        assert 'export' in pipeline_result
        
        # パフォーマンス検証
        assert processing_time < 60.0, f"パイプライン処理時間が長すぎ: {processing_time:.2f}秒"
        
        # 出力ファイル確認
        exported_files = list(pipeline_config['exported_data_dir'].glob('*.json'))
        assert len(exported_files) > 0, "出力ファイルが生成されていない"
        
        # ログファイル確認
        log_files = list(pipeline_config['logs_dir'].glob('*.log'))
        assert len(log_files) > 0, "ログファイルが生成されていない"
        
        # 品質レポート確認
        quality_reports = list(pipeline_config['logs_dir'].glob('quality_report_*.json'))
        assert len(quality_reports) > 0, "品質レポートが生成されていない"
        
        # 最終データ品質確認
        with open(exported_files[0], 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        assert 'videos' in final_data
        assert 'users' in final_data
        assert 'metadata' in final_data
        
        # メタデータ確認
        metadata = final_data['metadata']
        assert 'processing_time' in metadata
        assert 'total_records' in metadata
        assert 'quality_score' in metadata
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_pipeline_error_recovery(self, pipeline_config):
        """パイプラインエラー回復テスト"""
        pipeline_manager = PipelineManager(config=pipeline_config)
        
        # 無効なデータファイルでのテスト
        invalid_data_file = pipeline_config['raw_data_dir'] / 'invalid_data.json'
        with open(invalid_data_file, 'w') as f:
            f.write('{ invalid json content')
        
        # エラー回復テスト
        try:
            result = pipeline_manager.run_full_pipeline(
                source_path=str(invalid_data_file),
                enable_error_recovery=True
            )
            
            # エラーが適切にハンドリングされていることを確認
            assert result['success'] == False
            assert 'error' in result
            assert 'recovery_attempted' in result
            
        except Exception as e:
            # 例外がキャッチされた場合も、適切なエラー情報があることを確認
            assert isinstance(e, (json.JSONDecodeError, ValueError))
        
        # エラーログの生成確認
        error_logs = list(pipeline_config['logs_dir'].glob('error_*.log'))
        assert len(error_logs) > 0, "エラーログが生成されていない"
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_concurrent_pipeline_processing(self, sample_raw_data, pipeline_config):
        """並行パイプライン処理テスト"""
        pipeline_manager = PipelineManager(config=pipeline_config)
        
        # 複数のデータファイルを作成
        data_files = []
        for i in range(3):
            data_file = pipeline_config['raw_data_dir'] / f'concurrent_test_{i}.json'
            
            # 各ファイルに異なるデータを設定
            test_data = sample_raw_data.copy()
            for j, video in enumerate(test_data['videos']):
                video['external_id'] = f'concurrent_{i}_{j}'
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            data_files.append(str(data_file))
        
        # 並行処理実行
        start_time = datetime.now()
        
        results = pipeline_manager.run_concurrent_pipelines(
            source_paths=data_files,
            max_workers=2
        )
        
        end_time = datetime.now()
        concurrent_time = (end_time - start_time).total_seconds()
        
        # 結果検証
        assert len(results) == 3
        for result in results:
            assert result['success'] == True
        
        # 並行処理効率の確認（シーケンシャル処理より高速であることを期待）
        assert concurrent_time < 180.0, f"並行処理時間が長すぎ: {concurrent_time:.2f}秒"
        
        # 出力ファイルの重複チェック
        exported_files = list(pipeline_config['exported_data_dir'].glob('*.json'))
        external_ids = set()
        
        for file_path in exported_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for video in data.get('videos', []):
                external_id = video.get('external_id')
                if external_id:
                    assert external_id not in external_ids, f"重複するexternal_id: {external_id}"
                    external_ids.add(external_id)
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_pipeline_configuration_validation(self, temp_pipeline_dir):
        """パイプライン設定検証テスト"""
        # 無効な設定でのテスト
        invalid_configs = [
            # 存在しないディレクトリ
            {
                'data_dir': '/non/existent/directory',
                'batch_size': 100,
                'quality_threshold': 0.8
            },
            # 無効なバッチサイズ
            {
                'data_dir': temp_pipeline_dir,
                'batch_size': 0,
                'quality_threshold': 0.8
            },
            # 無効な品質閾値
            {
                'data_dir': temp_pipeline_dir,
                'batch_size': 100,
                'quality_threshold': 1.5
            }
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, FileNotFoundError, AssertionError)):
                pipeline_manager = PipelineManager(config=config)
                pipeline_manager.validate_configuration()
        
        # 有効な設定のテスト
        valid_config = {
            'data_dir': temp_pipeline_dir,
            'batch_size': 100,
            'quality_threshold': 0.8,
            'max_workers': 2,
            'enable_validation': True
        }
        
        pipeline_manager = PipelineManager(config=valid_config)
        validation_result = pipeline_manager.validate_configuration()
        
        assert validation_result['valid'] == True
        assert 'warnings' in validation_result
        assert 'recommendations' in validation_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
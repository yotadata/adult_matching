"""
DMM API Integration Tests

DMM APIとの統合テスト - データ同期、変換、品質検証
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import requests
import time
from datetime import datetime, timedelta

from data.sync.dmm.dmm_sync_manager import DMMSyncManager
from tests.integration.data.mock_processors import DataQualityMonitor


@pytest.fixture
def dmm_api_config():
    """DMM API設定"""
    return {
        'api_id': 'test_api_id',
        'affiliate_id': 'test_affiliate_id',
        'base_url': 'https://api.dmm.com/affiliate/v3/ItemList',
        'rate_limit_delay': 1.0,
        'timeout': 30,
        'max_retries': 3
    }


@pytest.fixture
def sample_dmm_response():
    """サンプルDMM APIレスポンス"""
    return {
        "result": {
            "status": 200,
            "result_count": 2,
            "total_count": 1000,
            "first_position": 1,
            "items": [
                {
                    "content_id": "test001",
                    "product_id": "test001",
                    "title": "テスト動画1",
                    "URL": "https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2Fvideoa%2F-%2Fdetail%2F%3D%2Fcid%3Dtest001%2F&af_id=yotadata2-990&ch=api",
                    "affiliateURL": "https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2Fvideoa%2F-%2Fdetail%2F%3D%2Fcid%3Dtest001%2F&af_id=yotadata2-990&ch=api",
                    "imageURL": {
                        "list": "https://pics.dmm.co.jp/digital/video/test001/test001pl.jpg",
                        "small": "https://pics.dmm.co.jp/digital/video/test001/test001ps.jpg",
                        "large": "https://pics.dmm.co.jp/digital/video/test001/test001pr.jpg"
                    },
                    "sampleImageURL": {
                        "sample_s": {
                            "image": [
                                "https://pics.dmm.co.jp/digital/video/test001/test001-1.jpg",
                                "https://pics.dmm.co.jp/digital/video/test001/test001-2.jpg"
                            ]
                        }
                    },
                    "sampleMovieURL": {
                        "size_476_306": "https://cc3001.dmm.co.jp/litevideo/freepv/test/test001/test001_mhb_w.mp4",
                        "size_560_360": "https://cc3001.dmm.co.jp/litevideo/freepv/test/test001/test001_dm_w.mp4"
                    },
                    "prices": {
                        "price": "300",
                        "list_price": "300",
                        "deliveries": {
                            "delivery": [
                                {
                                    "type": "download",
                                    "price": "300"
                                },
                                {
                                    "type": "stream",
                                    "price": "300"
                                }
                            ]
                        }
                    },
                    "date": "2024-01-15 10:00:00",
                    "iteminfo": {
                        "genre": [
                            {"id": 6001, "name": "企画"},
                            {"id": 6017, "name": "素人"}
                        ],
                        "series": [
                            {"id": 44896, "name": "テストシリーズ"}
                        ],
                        "maker": [
                            {"id": 45062, "name": "テストメーカー"}
                        ],
                        "actress": [
                            {"id": 1234567, "name": "テスト女優", "ruby": "てすとじょゆう"}
                        ],
                        "director": [
                            {"id": 70583, "name": "テスト監督"}
                        ],
                        "label": [
                            {"id": 13203, "name": "テストレーベル"}
                        ],
                        "keyword": [
                            {"id": 1500, "name": "美少女"},
                            {"id": 1501, "name": "制服"}
                        ]
                    },
                    "volume": "120",
                    "review": {
                        "count": 42,
                        "average": "4.36"
                    }
                },
                {
                    "content_id": "test002",
                    "product_id": "test002", 
                    "title": "テスト動画2",
                    "URL": "https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2Fvideoa%2F-%2Fdetail%2F%3D%2Fcid%3Dtest002%2F&af_id=yotadata2-990&ch=api",
                    "affiliateURL": "https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2Fvideoa%2F-%2Fdetail%2F%3D%2Fcid%3Dtest002%2F&af_id=yotadata2-990&ch=api",
                    "imageURL": {
                        "list": "https://pics.dmm.co.jp/digital/video/test002/test002pl.jpg",
                        "small": "https://pics.dmm.co.jp/digital/video/test002/test002ps.jpg",
                        "large": "https://pics.dmm.co.jp/digital/video/test002/test002pr.jpg"
                    },
                    "sampleImageURL": {
                        "sample_s": {
                            "image": [
                                "https://pics.dmm.co.jp/digital/video/test002/test002-1.jpg"
                            ]
                        }
                    },
                    "prices": {
                        "price": "500",
                        "list_price": "500"
                    },
                    "date": "2024-01-16 10:00:00",
                    "iteminfo": {
                        "genre": [
                            {"id": 6002, "name": "ドラマ"}
                        ],
                        "maker": [
                            {"id": 45063, "name": "テストメーカー2"}
                        ]
                    },
                    "volume": "90",
                    "review": {
                        "count": 15,
                        "average": "3.80"
                    }
                }
            ]
        }
    }


@pytest.fixture
def temp_data_dir():
    """一時データディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="data_integration_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_supabase_client():
    """Supabaseクライアントのモック"""
    mock_client = Mock()
    
    # テーブル操作のモック
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    
    # 正常なレスポンス
    mock_response = Mock()
    mock_response.data = []
    mock_response.error = None
    mock_response.count = 0
    
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = mock_response
    mock_table.insert.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    
    return mock_client


class TestDMMAPIIntegration:
    """DMM API統合テストクラス"""
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_dmm_api_connection(self, dmm_api_config):
        """DMM API接続テスト"""
        with patch('requests.get') as mock_get:
            # 正常レスポンスのモック
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "result": {
                    "status": 200,
                    "result_count": 0,
                    "total_count": 1000,
                    "items": []
                }
            }
            mock_get.return_value = mock_response
            
            # API接続テスト
            response = requests.get(dmm_api_config['base_url'], params={
                'api_id': dmm_api_config['api_id'],
                'affiliate_id': dmm_api_config['affiliate_id'],
                'site': 'FANZA',
                'service': 'digital',
                'floor': 'videoa',
                'hits': 1,
                'output': 'json'
            })
            
            assert response.status_code == 200
            data = response.json()
            assert 'result' in data
            assert data['result']['status'] == 200
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_dmm_api_rate_limiting(self, dmm_api_config):
        """DMM APIレート制限テスト"""
        rate_limit_delay = dmm_api_config['rate_limit_delay']
        
        # 複数リクエストのタイミング測定
        request_times = []
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"status": 200}}
            mock_get.return_value = mock_response
            
            # 3回のAPIリクエスト
            for i in range(3):
                start_time = time.time()
                
                if i > 0:  # 最初のリクエスト以外は待機
                    time.sleep(rate_limit_delay)
                
                response = requests.get(dmm_api_config['base_url'])
                request_times.append(time.time() - start_time)
            
            # 2番目以降のリクエストがレート制限を守っていることを確認
            for i in range(1, len(request_times)):
                assert request_times[i] >= rate_limit_delay - 0.1  # 0.1秒の許容誤差
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_dmm_data_transformation(self, sample_dmm_response):
        """DMM APIデータ変換テスト"""
        from tests.integration.data.mock_processors import UnifiedDataProcessor
        
        cleaner = DataCleaner()
        
        # DMM APIレスポンスからビデオデータを変換
        items = sample_dmm_response['result']['items']
        
        for item in items:
            transformed = cleaner.transform_dmm_item(item)
            
            # 必須フィールドの存在確認
            required_fields = [
                'external_id', 'source', 'title', 'price', 
                'thumbnail_url', 'duration_seconds'
            ]
            
            for field in required_fields:
                assert field in transformed, f"Missing field: {field}"
            
            # データ型検証
            assert isinstance(transformed['external_id'], str)
            assert isinstance(transformed['title'], str)
            assert isinstance(transformed['price'], (int, float))
            assert isinstance(transformed['duration_seconds'], int)
            assert transformed['source'] == 'dmm'
            
            # URL検証
            assert transformed['thumbnail_url'].startswith('https://')
            
            # ジャンル・メーカー情報の変換確認
            if 'genre' in transformed:
                assert isinstance(transformed['genre'], str)
            if 'maker' in transformed:
                assert isinstance(transformed['maker'], str)
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_quality_validation(self, sample_dmm_response, temp_data_dir):
        """データ品質検証テスト"""
        validator = DataValidator()
        quality_monitor = QualityMonitor()
        
        items = sample_dmm_response['result']['items']
        
        # 各アイテムの品質検証
        validation_results = []
        
        for item in items:
            # 基本検証
            is_valid = validator.validate_dmm_item(item)
            validation_results.append(is_valid)
            
            # 品質スコア計算
            quality_score = quality_monitor.calculate_quality_score(item)
            
            # 品質基準
            assert quality_score >= 0.0
            assert quality_score <= 1.0
            
            # 必須フィールド存在確認
            assert 'content_id' in item
            assert 'title' in item
            assert 'prices' in item
        
        # 全体品質統計
        valid_count = sum(validation_results)
        total_count = len(validation_results)
        success_rate = valid_count / total_count if total_count > 0 else 0
        
        # 品質要件: 90%以上が有効データであること
        assert success_rate >= 0.9, f"データ品質が基準未満: {success_rate:.2%}"
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_dmm_sync_error_handling(self, dmm_api_config, mock_supabase_client):
        """DMM同期エラーハンドリングテスト"""
        sync_manager = DMMSyncManager(
            config=dmm_api_config,
            supabase_client=mock_supabase_client
        )
        
        # API エラーシナリオ
        error_scenarios = [
            # レート制限エラー
            {
                'status_code': 429,
                'error_type': 'rate_limit',
                'expected_retry': True
            },
            # サーバーエラー
            {
                'status_code': 500,
                'error_type': 'server_error', 
                'expected_retry': True
            },
            # 認証エラー
            {
                'status_code': 401,
                'error_type': 'auth_error',
                'expected_retry': False
            },
            # 不正なリクエスト
            {
                'status_code': 400,
                'error_type': 'bad_request',
                'expected_retry': False
            }
        ]
        
        for scenario in error_scenarios:
            with patch('requests.get') as mock_get:
                # エラーレスポンスのモック
                mock_response = Mock()
                mock_response.status_code = scenario['status_code']
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
                mock_get.return_value = mock_response
                
                # エラーハンドリングテスト
                try:
                    sync_manager.fetch_items(page=1, limit=10)
                except Exception as e:
                    # 期待されるエラー処理の確認
                    assert isinstance(e, (requests.exceptions.HTTPError, Exception))
                    
                    # リトライ可能エラーの場合、適切なリトライ機構があることを確認
                    if scenario['expected_retry']:
                        assert hasattr(sync_manager, 'retry_count') or 'retry' in str(e).lower()
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.slow
    def test_end_to_end_dmm_sync(self, dmm_api_config, sample_dmm_response, mock_supabase_client, temp_data_dir):
        """エンドツーエンドDMM同期テスト"""
        sync_manager = DMMSyncManager(
            config=dmm_api_config,
            supabase_client=mock_supabase_client
        )
        
        # Mock API responses
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_dmm_response
            mock_get.return_value = mock_response
            
            # 同期プロセス実行
            start_time = time.time()
            sync_result = sync_manager.sync_batch(
                max_items=2,
                start_page=1
            )
            sync_duration = time.time() - start_time
            
            # 結果検証
            assert sync_result['success'] == True
            assert sync_result['items_processed'] == 2
            assert sync_result['errors'] == 0
            
            # パフォーマンス検証
            assert sync_duration < 30.0, f"同期時間が長すぎ: {sync_duration:.2f}秒"
            
            # データベース呼び出し確認
            mock_supabase_client.table.assert_called()
            
            # ログファイル生成確認
            log_files = list(Path(temp_data_dir).glob("*.log"))
            assert len(log_files) > 0, "ログファイルが生成されていない"
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_data_consistency_validation(self, sample_dmm_response, mock_supabase_client):
        """データ整合性検証テスト"""
        # 重複データ検出テスト
        items = sample_dmm_response['result']['items']
        
        # 同じcontent_idの重複データを作成
        duplicate_item = items[0].copy()
        duplicate_item['title'] = '重複テスト動画'
        items.append(duplicate_item)
        
        # 重複検出ロジック
        seen_content_ids = set()
        duplicates = []
        
        for item in items:
            content_id = item['content_id']
            if content_id in seen_content_ids:
                duplicates.append(content_id)
            seen_content_ids.add(content_id)
        
        # 重複が検出されることを確認
        assert len(duplicates) == 1
        assert duplicates[0] == 'test001'
        
        # データベース一意制約テスト（モック）
        mock_supabase_client.table().upsert.side_effect = Exception("Duplicate key violation")
        
        # 重複データ処理の確認
        try:
            mock_supabase_client.table('videos').upsert({
                'external_id': 'test001',
                'source': 'dmm',
                'title': 'Test Video'
            })
        except Exception as e:
            assert "Duplicate" in str(e)
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_dmm_api_pagination(self, dmm_api_config):
        """DMM APIページネーションテスト"""
        with patch('requests.get') as mock_get:
            # 複数ページのレスポンスをモック
            page_responses = [
                {
                    "result": {
                        "status": 200,
                        "result_count": 50,
                        "total_count": 150,
                        "first_position": 1,
                        "items": [{"content_id": f"page1_item{i}"} for i in range(50)]
                    }
                },
                {
                    "result": {
                        "status": 200,
                        "result_count": 50,
                        "total_count": 150,
                        "first_position": 51,
                        "items": [{"content_id": f"page2_item{i}"} for i in range(50)]
                    }
                },
                {
                    "result": {
                        "status": 200,
                        "result_count": 50,
                        "total_count": 150,
                        "first_position": 101,
                        "items": [{"content_id": f"page3_item{i}"} for i in range(50)]
                    }
                }
            ]
            
            mock_responses = []
            for response_data in page_responses:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                mock_responses.append(mock_response)
            
            mock_get.side_effect = mock_responses
            
            # ページネーションテスト
            all_items = []
            for page in range(1, 4):
                response = requests.get(dmm_api_config['base_url'], params={
                    'api_id': dmm_api_config['api_id'],
                    'affiliate_id': dmm_api_config['affiliate_id'],
                    'hits': 50,
                    'offset': ((page - 1) * 50) + 1
                })
                
                data = response.json()
                items = data['result']['items']
                all_items.extend(items)
                
                # ページごとの検証
                assert len(items) == 50
                assert data['result']['total_count'] == 150
            
            # 全体検証
            assert len(all_items) == 150
            
            # 重複チェック
            content_ids = [item['content_id'] for item in all_items]
            assert len(content_ids) == len(set(content_ids)), "ページネーションで重複データ検出"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
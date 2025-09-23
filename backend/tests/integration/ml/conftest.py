"""
ML Integration Tests Configuration

ML統合テスト設定
共通フィクスチャとテスト環境設定
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from backend.ml.config import TrainingConfig
from backend.ml.preprocessing.features.feature_processor import FeatureConfig


@pytest.fixture(scope="session")
def ml_test_environment():
    """ML統合テスト環境（セッションレベル）"""
    # テスト用一時ディレクトリ作成
    base_temp_dir = tempfile.mkdtemp(prefix="ml_integration_test_session_")

    # ディレクトリ構造作成
    directories = [
        "models", "data", "artifacts", "logs", "exports",
        "preprocessors", "features", "embeddings", "checkpoints",
        "tensorboard", "metrics", "deployments"
    ]

    env = {"base_dir": Path(base_temp_dir)}

    for dir_name in directories:
        dir_path = Path(base_temp_dir) / dir_name
        dir_path.mkdir(exist_ok=True)
        env[f"{dir_name}_dir"] = dir_path

    # 環境変数設定
    os.environ['ML_TEST_MODE'] = 'true'
    os.environ['ML_TEST_BASE_DIR'] = str(base_temp_dir)

    yield env

    # クリーンアップ
    shutil.rmtree(base_temp_dir, ignore_errors=True)
    os.environ.pop('ML_TEST_MODE', None)
    os.environ.pop('ML_TEST_BASE_DIR', None)


@pytest.fixture
def standard_config():
    """標準テスト設定"""
    return TrainingConfig(
        user_embedding_dim=768,
        item_embedding_dim=768,
        user_hidden_units=[512, 256, 128],
        item_hidden_units=[512, 256, 128],
        batch_size=64,
        epochs=3,
        learning_rate=0.001,
        validation_split=0.2,
        early_stopping_patience=2,
        save_best_only=True,
        reduce_lr_on_plateau=True
    )


@pytest.fixture
def fast_test_config():
    """高速テスト用設定（開発用）"""
    return TrainingConfig(
        user_embedding_dim=64,
        item_embedding_dim=64,
        user_hidden_units=[32, 16],
        item_hidden_units=[32, 16],
        batch_size=32,
        epochs=1,
        learning_rate=0.01,
        validation_split=0.1
    )


@pytest.fixture
def feature_test_config():
    """特徴量処理テスト用設定"""
    return FeatureConfig(
        target_dimension=768,
        user_features=[
            'age', 'preferences', 'activity_level',
            'registration_date', 'premium_user'
        ],
        item_features=[
            'genre', 'duration', 'quality_score',
            'popularity_score', 'features', 'tags'
        ],
        categorical_features=[
            'genre', 'age_group', 'user_type'
        ],
        numerical_features=[
            'age', 'duration', 'quality_score',
            'activity_level', 'popularity_score'
        ],
        embedding_features=[
            'preferences', 'features', 'tags'
        ]
    )


@pytest.fixture
def sample_small_dataset():
    """小規模テストデータセット"""
    np.random.seed(42)

    users = pd.DataFrame([
        {
            'user_id': f'user_{i}',
            'age': np.random.randint(18, 60),
            'preferences': np.random.random(10).tolist(),
            'activity_level': np.random.random(),
            'registration_date': f"2024-{np.random.randint(1, 13):02d}-01",
            'premium_user': np.random.choice([True, False], p=[0.3, 0.7])
        }
        for i in range(50)
    ])

    items = pd.DataFrame([
        {
            'item_id': f'video_{i}',
            'genre': np.random.choice(['action', 'romance', 'comedy', 'drama']),
            'duration': np.random.randint(60, 180),
            'quality_score': np.random.random(),
            'popularity_score': np.random.random(),
            'features': np.random.random(15).tolist(),
            'tags': np.random.choice(['tag1', 'tag2', 'tag3'], size=3).tolist()
        }
        for i in range(100)
    ])

    interactions = pd.DataFrame([
        {
            'user_id': f'user_{np.random.randint(0, 50)}',
            'item_id': f'video_{np.random.randint(0, 100)}',
            'rating': np.random.choice([0, 1], p=[0.3, 0.7]),
            'timestamp': f"2024-01-{np.random.randint(1, 30):02d}",
            'interaction_type': np.random.choice(['like', 'view', 'share'])
        }
        for i in range(500)
    ])

    return {
        'users': users,
        'items': items,
        'interactions': interactions
    }


@pytest.fixture
def sample_medium_dataset():
    """中規模テストデータセット"""
    np.random.seed(42)

    users = pd.DataFrame([
        {
            'user_id': f'user_{i}',
            'age': np.random.randint(18, 65),
            'preferences': np.random.random(15).tolist(),
            'activity_level': np.random.random(),
            'registration_date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            'premium_user': np.random.choice([True, False], p=[0.25, 0.75])
        }
        for i in range(1000)
    ])

    items = pd.DataFrame([
        {
            'item_id': f'video_{i}',
            'genre': np.random.choice(['action', 'romance', 'comedy', 'drama', 'thriller', 'sci-fi']),
            'duration': np.random.randint(30, 240),
            'quality_score': np.random.random(),
            'popularity_score': np.random.random(),
            'features': np.random.random(20).tolist(),
            'tags': np.random.choice(['tag1', 'tag2', 'tag3', 'tag4', 'tag5'], size=4).tolist()
        }
        for i in range(5000)
    ])

    interactions = pd.DataFrame([
        {
            'user_id': f'user_{np.random.randint(0, 1000)}',
            'item_id': f'video_{np.random.randint(0, 5000)}',
            'rating': np.random.choice([0, 1], p=[0.2, 0.8]),
            'timestamp': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            'interaction_type': np.random.choice(['like', 'view', 'share', 'download'])
        }
        for i in range(50000)
    ])

    return {
        'users': users,
        'items': items,
        'interactions': interactions
    }


@pytest.fixture
def mock_database():
    """モックデータベース"""
    class MockDatabase:
        def __init__(self):
            self.data = {}

        def insert(self, table: str, data: Dict[str, Any]) -> bool:
            if table not in self.data:
                self.data[table] = []
            self.data[table].append(data)
            return True

        def select(self, table: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            if table not in self.data:
                return []

            result = self.data[table]

            if filters:
                for key, value in filters.items():
                    result = [row for row in result if row.get(key) == value]

            return result

        def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> bool:
            if table not in self.data:
                return False

            for row in self.data[table]:
                match = True
                for key, value in filters.items():
                    if row.get(key) != value:
                        match = False
                        break

                if match:
                    row.update(updates)

            return True

        def delete(self, table: str, filters: Dict[str, Any]) -> bool:
            if table not in self.data:
                return False

            original_count = len(self.data[table])
            self.data[table] = [
                row for row in self.data[table]
                if not all(row.get(k) == v for k, v in filters.items())
            ]

            return len(self.data[table]) < original_count

    return MockDatabase()


@pytest.fixture
def mock_supabase_client():
    """モックSupabaseクライアント"""
    class MockSupabaseClient:
        def __init__(self):
            self.data = {}

        def table(self, table_name: str):
            return MockTable(table_name, self.data)

    class MockTable:
        def __init__(self, table_name: str, data_storage: Dict):
            self.table_name = table_name
            self.data_storage = data_storage

        def select(self, columns: str = "*"):
            return MockQuery(self.table_name, self.data_storage)

        def insert(self, data: Dict[str, Any]):
            if self.table_name not in self.data_storage:
                self.data_storage[self.table_name] = []
            self.data_storage[self.table_name].append(data)
            return MockResponse([data])

        def update(self, data: Dict[str, Any]):
            return MockQuery(self.table_name, self.data_storage, update_data=data)

        def delete(self):
            return MockQuery(self.table_name, self.data_storage, delete_mode=True)

    class MockQuery:
        def __init__(self, table_name: str, data_storage: Dict, update_data: Dict = None, delete_mode: bool = False):
            self.table_name = table_name
            self.data_storage = data_storage
            self.update_data = update_data
            self.delete_mode = delete_mode
            self.filters = {}

        def eq(self, column: str, value: Any):
            self.filters[column] = value
            return self

        def limit(self, count: int):
            self.limit_count = count
            return self

        def execute(self):
            if self.table_name not in self.data_storage:
                return MockResponse([])

            data = self.data_storage[self.table_name]

            # フィルタ適用
            if self.filters:
                data = [
                    row for row in data
                    if all(row.get(k) == v for k, v in self.filters.items())
                ]

            # 操作実行
            if self.delete_mode:
                self.data_storage[self.table_name] = [
                    row for row in self.data_storage[self.table_name]
                    if not all(row.get(k) == v for k, v in self.filters.items())
                ]
                return MockResponse([])

            if self.update_data:
                for row in self.data_storage[self.table_name]:
                    if all(row.get(k) == v for k, v in self.filters.items()):
                        row.update(self.update_data)
                return MockResponse(data)

            # 制限適用
            if hasattr(self, 'limit_count'):
                data = data[:self.limit_count]

            return MockResponse(data)

    class MockResponse:
        def __init__(self, data: List[Dict[str, Any]]):
            self.data = data

    return MockSupabaseClient()


@pytest.fixture(autouse=True)
def ml_test_patches():
    """ML統合テスト用パッチ（自動適用）"""
    patches = []

    # 長時間処理のモック
    mock_training = Mock(return_value={
        'success': True,
        'final_loss': 0.234,
        'training_time_minutes': 5.0,
        'epochs_completed': 3,
        'training_history': [
            {'epoch': 1, 'loss': 0.456, 'val_loss': 0.478},
            {'epoch': 2, 'loss': 0.345, 'val_loss': 0.367},
            {'epoch': 3, 'loss': 0.234, 'val_loss': 0.248}
        ]
    })

    # 必要に応じてパッチを追加
    # patches.append(patch('backend.ml.training.trainers.unified_two_tower_trainer.UnifiedTwoTowerTrainer.train', mock_training))

    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()


# カスタムマーカー
def pytest_configure(config):
    """カスタムマーカー設定"""
    config.addinivalue_line("markers", "ml: ML related tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: Tests requiring trained model")


# テスト実行前のセットアップ
def pytest_sessionstart(session):
    """テストセッション開始時処理"""
    print("\n🚀 Starting ML Integration Test Session...")
    print("Setting up test environment...")


def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時処理"""
    print("\n✅ ML Integration Test Session Complete")
    print(f"Exit status: {exitstatus}")


# テストファイル毎のセットアップ
def pytest_runtest_setup(item):
    """各テスト実行前処理"""
    if item.get_closest_marker("requires_gpu"):
        # GPU要求テストのチェック
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("TensorFlow not available")

    if item.get_closest_marker("slow"):
        # 環境変数でスロー テストスキップ
        if os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true":
            pytest.skip("Slow test skipped")


# テスト結果レポート
def pytest_runtest_logreport(report):
    """テスト結果レポート"""
    if report.when == "call":
        if report.passed:
            print(f"✅ {report.nodeid}")
        elif report.failed:
            print(f"❌ {report.nodeid}")
        elif report.skipped:
            print(f"⏭️  {report.nodeid}")
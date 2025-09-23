"""
Basic ML Integration Tests

MLパイプライン基本統合テスト
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from backend.ml.config import TrainingConfig
from backend.ml.training.trainers.unified_two_tower_trainer import UnifiedTwoTowerTrainer


@pytest.fixture
def simple_config():
    """シンプルなトレーニング設定"""
    return TrainingConfig(
        user_embedding_dim=64,
        item_embedding_dim=64,
        user_hidden_units=[32, 16],
        item_hidden_units=[32, 16],
        batch_size=32,
        epochs=2,
        learning_rate=0.001
    )


@pytest.fixture
def temp_test_dir():
    """一時テストディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="ml_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBasicMLIntegration:
    """基本MLパイプライン統合テスト"""
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_ml_trainer_initialization(self, simple_config, temp_test_dir):
        """MLトレーナー初期化テスト"""
        trainer = UnifiedTwoTowerTrainer(
            config=simple_config,
            model_save_dir=temp_test_dir,
            experiment_name="init_test"
        )
        
        assert trainer is not None
        assert trainer.config == simple_config
        assert trainer.experiment_name == "init_test"
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_config_loading(self):
        """設定読み込みテスト"""
        from backend.ml.config import TrainingConfig
        
        config = TrainingConfig()
        
        # デフォルト値確認
        assert config.user_embedding_dim > 0
        assert config.item_embedding_dim > 0
        assert isinstance(config.user_hidden_units, list)
        assert isinstance(config.item_hidden_units, list)
        assert config.batch_size > 0
        assert config.epochs > 0
        assert 0 < config.learning_rate < 1
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_model_utilities(self):
        """モデルユーティリティテスト"""
        from backend.ml.utils.model_utils import ModelUtils
        
        # 特徴量正規化テスト
        features = np.random.random((100, 10))
        normalized = ModelUtils.normalize_features(features, method="standard")
        
        assert normalized.shape == features.shape
        assert not np.any(np.isnan(normalized))
        
        # 類似度計算テスト
        user_embeddings = np.random.random((10, 64))
        item_embeddings = np.random.random((10, 64))
        
        similarities = ModelUtils.calculate_similarity(user_embeddings, item_embeddings)
        
        assert len(similarities) == 10
        assert all(-1 <= s <= 1 for s in similarities)
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_deployment_tools(self, temp_test_dir):
        """デプロイメントツールテスト"""
        from backend.ml.deployment.model_converter import ModelConverter
        from backend.ml.deployment.model_deployer import ModelDeployer
        
        converter = ModelConverter()
        deployer = ModelDeployer()
        
        assert converter is not None
        assert deployer is not None
        
        # Web配信パッケージ作成テスト
        web_package_path = Path(temp_test_dir) / "web_package"
        
        # ダミートレーナー
        class DummyTrainer:
            pass
        
        dummy_trainer = DummyTrainer()
        
        result = deployer.create_web_package(
            model_trainer=dummy_trainer,
            output_path=str(web_package_path),
            include_preprocessors=True
        )
        
        assert result['success'] == True
        assert web_package_path.exists()
        assert (web_package_path / "package.json").exists()
        assert (web_package_path / "inference.js").exists()
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_evaluation_metrics(self):
        """評価指標テスト"""
        from backend.ml.evaluation.metrics.model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # テストデータ
        true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        predictions = [0.8, 0.2, 0.9, 0.7, 0.1, 0.6, 0.3, 0.2, 0.8, 0.4]
        
        metrics = evaluator.calculate_metrics(true_labels, predictions)
        
        # 基本指標の存在確認
        required_metrics = ['auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
        
        # ランキング指標テスト
        ranking_metrics = evaluator.calculate_ranking_metrics(true_labels, predictions)
        
        assert 'precision_at_5' in ranking_metrics
        assert 'ndcg_at_5' in ranking_metrics
        assert all(0 <= v <= 1 for v in ranking_metrics.values())
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_version_management(self, temp_test_dir):
        """バージョン管理テスト"""
        from backend.ml.deployment.version_manager import ModelVersionManager
        
        version_manager = ModelVersionManager(base_path=temp_test_dir)
        
        # ダミーモデル
        class DummyModel:
            pass
        
        dummy_model = DummyModel()
        
        # バージョン登録
        result = version_manager.register_version(
            model_trainer=dummy_model,
            version_tag="v1.0.0",
            description="Test version",
            metrics={"accuracy": 0.85, "auc_roc": 0.78}
        )
        
        assert result['success'] == True
        
        # バージョン一覧確認
        versions = version_manager.list_versions()
        assert len(versions) == 1
        assert versions[0]['version'] == "v1.0.0"
        
        # 特定バージョン取得
        version_data = version_manager.get_version("v1.0.0")
        assert version_data is not None
        assert version_data['version'] == "v1.0.0"
        assert version_data['metrics']['accuracy'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
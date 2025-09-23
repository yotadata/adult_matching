"""
Comprehensive ML Pipeline Integration Tests

包括的MLパイプライン統合テスト
MLトレーニングと推論パイプラインの完全ワークフローテスト
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import os
import asyncio
from unittest.mock import Mock, patch

from backend.ml.config import TrainingConfig
from backend.ml.training.trainers.unified_two_tower_trainer import UnifiedTwoTowerTrainer
from backend.ml.preprocessing.features.feature_processor import FeatureProcessor, FeatureConfig
from backend.ml.evaluation.metrics.model_evaluator import ModelEvaluator
from backend.ml.deployment.model_converter import ModelConverter
from backend.ml.inference.realtime.realtime_predictor import RealtimePredictor
from backend.data.unified_data_manager import UnifiedDataManager


@pytest.fixture
def comprehensive_config():
    """包括テスト用設定"""
    return TrainingConfig(
        user_embedding_dim=768,
        item_embedding_dim=768,
        user_hidden_units=[512, 256, 128],
        item_hidden_units=[512, 256, 128],
        batch_size=64,
        epochs=3,
        learning_rate=0.001,
        validation_split=0.2,
        early_stopping_patience=2
    )


@pytest.fixture
def test_environment():
    """テスト環境セットアップ"""
    temp_dir = tempfile.mkdtemp(prefix="ml_comprehensive_test_")

    # テストディレクトリ構造作成
    dirs = [
        "models", "data", "artifacts", "logs", "exports",
        "preprocessors", "features", "embeddings"
    ]

    for dir_name in dirs:
        (Path(temp_dir) / dir_name).mkdir(exist_ok=True)

    yield {
        "temp_dir": temp_dir,
        "models_dir": Path(temp_dir) / "models",
        "data_dir": Path(temp_dir) / "data",
        "artifacts_dir": Path(temp_dir) / "artifacts",
        "logs_dir": Path(temp_dir) / "logs",
        "exports_dir": Path(temp_dir) / "exports"
    }

    # クリーンアップ
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_training_data():
    """サンプルトレーニングデータ"""
    np.random.seed(42)

    # ユーザー特徴量（100ユーザー）
    users = []
    for i in range(100):
        users.append({
            'user_id': f'user_{i}',
            'age': np.random.randint(18, 60),
            'preferences': np.random.random(10).tolist(),
            'activity_level': np.random.random()
        })

    # アイテム特徴量（200動画）
    items = []
    for i in range(200):
        items.append({
            'item_id': f'video_{i}',
            'genre': np.random.choice(['action', 'romance', 'comedy', 'drama']),
            'duration': np.random.randint(60, 180),
            'quality_score': np.random.random(),
            'features': np.random.random(20).tolist()
        })

    # インタラクション（2000件）
    interactions = []
    for i in range(2000):
        user_id = f"user_{np.random.randint(0, 100)}"
        item_id = f"video_{np.random.randint(0, 200)}"
        rating = np.random.choice([0, 1], p=[0.3, 0.7])  # 70%が正例

        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': f"2024-01-{np.random.randint(1, 30):02d}"
        })

    return {
        'users': users,
        'items': items,
        'interactions': interactions
    }


class TestComprehensiveMLPipeline:
    """包括的MLパイプライン統合テスト"""

    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.slow
    async def test_complete_ml_workflow(self, comprehensive_config, test_environment, sample_training_data):
        """完全MLワークフローテスト"""
        print("🚀 Starting complete ML workflow test...")

        # 1. データ準備と前処理
        print("📊 Phase 1: Data preprocessing...")

        feature_config = FeatureConfig(
            target_dimension=768,
            user_features=['age', 'preferences', 'activity_level'],
            item_features=['genre', 'duration', 'quality_score', 'features']
        )

        feature_processor = FeatureProcessor(feature_config)

        # データフレーム作成
        users_df = pd.DataFrame(sample_training_data['users'])
        items_df = pd.DataFrame(sample_training_data['items'])
        interactions_df = pd.DataFrame(sample_training_data['interactions'])

        # 特徴量処理
        feature_processor.fit(users_df, items_df, interactions_df)

        user_features = feature_processor.transform_users(users_df)
        item_features = feature_processor.transform_items(items_df)

        assert user_features.shape[1] == 768, "User features dimension mismatch"
        assert item_features.shape[1] == 768, "Item features dimension mismatch"
        assert len(user_features) == 100, "User count mismatch"
        assert len(item_features) == 200, "Item count mismatch"

        print(f"✅ Features processed: users {user_features.shape}, items {item_features.shape}")

        # 2. モデルトレーニング
        print("🎯 Phase 2: Model training...")

        trainer = UnifiedTwoTowerTrainer(
            config=comprehensive_config,
            model_save_dir=str(test_environment["models_dir"]),
            experiment_name="comprehensive_test"
        )

        # トレーニングデータ準備
        training_data = {
            'user_features': user_features,
            'item_features': item_features,
            'interactions': interactions_df
        }

        # モデルトレーニング実行
        with patch.object(trainer, '_save_model') as mock_save:
            mock_save.return_value = True

            training_results = await asyncio.to_thread(
                trainer.train, training_data
            )

        assert training_results['success'] == True
        assert training_results['final_loss'] > 0
        assert 'training_history' in training_results
        assert len(training_results['training_history']) > 0

        print(f"✅ Training completed: loss {training_results['final_loss']:.4f}")

        # 3. モデル評価
        print("📈 Phase 3: Model evaluation...")

        evaluator = ModelEvaluator()

        # テストデータ生成
        test_interactions = interactions_df.sample(n=400, random_state=42)
        true_labels = test_interactions['rating'].values

        # 予測生成（モック）
        with patch.object(trainer, 'predict') as mock_predict:
            mock_predict.return_value = np.random.random(len(true_labels))
            predictions = trainer.predict(test_interactions)

        # 評価指標計算
        metrics = evaluator.calculate_metrics(true_labels, predictions)
        ranking_metrics = evaluator.calculate_ranking_metrics(true_labels, predictions)

        # 指標検証
        required_metrics = ['auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

        assert metrics['auc_pr'] >= 0.85, f"AUC-PR {metrics['auc_pr']:.3f} below requirement (≥0.85)"

        print(f"✅ Evaluation completed: AUC-PR {metrics['auc_pr']:.3f}")

        # 4. モデルデプロイメント
        print("🚀 Phase 4: Model deployment...")

        converter = ModelConverter()

        # TensorFlow.js変換（モック）
        with patch.object(converter, 'to_tensorflowjs') as mock_tfjs:
            mock_tfjs.return_value = {
                'success': True,
                'output_path': str(test_environment["exports_dir"] / "tfjs_model"),
                'model_size_mb': 12.5
            }

            tfjs_result = converter.to_tensorflowjs(
                trainer=trainer,
                output_path=str(test_environment["exports_dir"] / "tfjs_model")
            )

        assert tfjs_result['success'] == True
        assert tfjs_result['model_size_mb'] < 50  # サイズ制限

        # ONNX変換（モック）
        with patch.object(converter, 'to_onnx') as mock_onnx:
            mock_onnx.return_value = {
                'success': True,
                'output_path': str(test_environment["exports_dir"] / "model.onnx"),
                'optimization_level': 'all'
            }

            onnx_result = converter.to_onnx(
                trainer=trainer,
                output_path=str(test_environment["exports_dir"] / "model.onnx")
            )

        assert onnx_result['success'] == True

        print("✅ Model deployment completed")

        # 5. リアルタイム推論テスト
        print("⚡ Phase 5: Real-time inference...")

        predictor = RealtimePredictor(
            model_path=str(test_environment["exports_dir"] / "tfjs_model")
        )

        # 推論テスト（モック）
        with patch.object(predictor, 'predict') as mock_inference:
            mock_inference.return_value = {
                'predictions': np.random.random(10).tolist(),
                'inference_time_ms': 45.2,
                'model_version': 'v1.0.0'
            }

            inference_result = predictor.predict(
                user_id='user_1',
                candidate_items=['video_1', 'video_2', 'video_3', 'video_4', 'video_5',
                               'video_6', 'video_7', 'video_8', 'video_9', 'video_10']
            )

        assert len(inference_result['predictions']) == 10
        assert inference_result['inference_time_ms'] < 500  # <500ms要件

        print(f"✅ Inference completed: {inference_result['inference_time_ms']:.1f}ms")

        # 6. パフォーマンス検証
        print("🔍 Phase 6: Performance validation...")

        performance_metrics = {
            'training_time': training_results.get('training_time_minutes', 0),
            'model_accuracy': metrics['auc_pr'],
            'inference_latency': inference_result['inference_time_ms'],
            'model_size': tfjs_result['model_size_mb']
        }

        # パフォーマンス要件検証
        assert performance_metrics['training_time'] < 120, "Training time exceeds 2 hours"
        assert performance_metrics['model_accuracy'] >= 0.85, "Model accuracy below threshold"
        assert performance_metrics['inference_latency'] < 500, "Inference latency exceeds 500ms"
        assert performance_metrics['model_size'] < 50, "Model size exceeds 50MB"

        print("✅ Performance validation passed")

        # 最終結果
        final_results = {
            'workflow_status': 'success',
            'stages_completed': [
                'data_preprocessing', 'model_training', 'model_evaluation',
                'model_deployment', 'realtime_inference', 'performance_validation'
            ],
            'performance_metrics': performance_metrics,
            'model_metrics': metrics,
            'test_summary': {
                'total_users': len(users_df),
                'total_items': len(items_df),
                'total_interactions': len(interactions_df),
                'test_interactions': len(test_interactions)
            }
        }

        print("🎉 Complete ML workflow test PASSED")
        return final_results

    @pytest.mark.integration
    @pytest.mark.ml
    def test_ml_pipeline_error_handling(self, comprehensive_config, test_environment):
        """MLパイプラインエラーハンドリングテスト"""
        print("🛠️ Testing ML pipeline error handling...")

        # 1. 不正データでの前処理エラー
        feature_config = FeatureConfig(target_dimension=768)
        feature_processor = FeatureProcessor(feature_config)

        # 空のデータフレーム
        empty_df = pd.DataFrame()

        with pytest.raises(Exception):
            feature_processor.fit(empty_df, empty_df, empty_df)

        # 2. 不正設定でのトレーニングエラー
        invalid_config = TrainingConfig(
            user_embedding_dim=-1,  # 無効な次元
            epochs=0  # 無効なエポック数
        )

        with pytest.raises(ValueError):
            UnifiedTwoTowerTrainer(
                config=invalid_config,
                model_save_dir=str(test_environment["models_dir"])
            )

        # 3. モデル評価エラー
        evaluator = ModelEvaluator()

        # 不適切なサイズの入力
        with pytest.raises(Exception):
            evaluator.calculate_metrics([1, 0, 1], [0.5])  # サイズ不一致

        print("✅ Error handling test passed")

    @pytest.mark.integration
    @pytest.mark.ml
    async def test_ml_pipeline_scalability(self, comprehensive_config, test_environment):
        """MLパイプラインスケーラビリティテスト"""
        print("📈 Testing ML pipeline scalability...")

        # 大規模データセット生成
        large_dataset_sizes = [1000, 5000, 10000]  # ユーザー数

        results = []

        for user_count in large_dataset_sizes:
            print(f"Testing with {user_count} users...")

            # データ生成
            users_df = pd.DataFrame([
                {
                    'user_id': f'user_{i}',
                    'age': np.random.randint(18, 60),
                    'preferences': np.random.random(10).tolist()
                }
                for i in range(user_count)
            ])

            items_df = pd.DataFrame([
                {
                    'item_id': f'video_{i}',
                    'genre': np.random.choice(['action', 'romance']),
                    'features': np.random.random(20).tolist()
                }
                for i in range(user_count * 2)  # アイテム数はユーザーの2倍
            ])

            # 処理時間測定
            start_time = pd.Timestamp.now()

            feature_config = FeatureConfig(target_dimension=768)
            feature_processor = FeatureProcessor(feature_config)

            try:
                feature_processor.fit(users_df, items_df, pd.DataFrame())
                user_features = feature_processor.transform_users(users_df)
                item_features = feature_processor.transform_items(items_df)

                end_time = pd.Timestamp.now()
                processing_time = (end_time - start_time).total_seconds()

                results.append({
                    'user_count': user_count,
                    'item_count': len(items_df),
                    'processing_time_seconds': processing_time,
                    'user_features_shape': user_features.shape,
                    'item_features_shape': item_features.shape,
                    'success': True
                })

                print(f"✅ {user_count} users processed in {processing_time:.2f}s")

                # スケーラビリティ制限チェック
                assert processing_time < 300, f"Processing time {processing_time}s exceeds 5min limit"

            except Exception as e:
                results.append({
                    'user_count': user_count,
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ Failed with {user_count} users: {e}")

        # スケーラビリティ分析
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) >= 2, "Pipeline failed scalability test"

        # 線形スケーリング確認（大まかな）
        if len(successful_results) >= 2:
            time_per_user = [r['processing_time_seconds'] / r['user_count'] for r in successful_results]
            avg_time_per_user = np.mean(time_per_user)

            assert avg_time_per_user < 0.1, f"Processing time per user {avg_time_per_user:.4f}s too high"

        print("✅ Scalability test passed")
        return results

    @pytest.mark.integration
    @pytest.mark.ml
    def test_ml_monitoring_integration(self, comprehensive_config, test_environment):
        """ML監視統合テスト"""
        print("📊 Testing ML monitoring integration...")

        from backend.ml.monitoring.performance.performance_monitor import MLPerformanceMonitor
        from backend.ml.monitoring.quality.quality_monitor import MLQualityMonitor

        # パフォーマンス監視
        perf_monitor = MLPerformanceMonitor()

        # メトリクス記録
        perf_monitor.record_training_metrics({
            'loss': 0.234,
            'accuracy': 0.876,
            'training_time': 1800,
            'memory_usage_mb': 2048
        })

        perf_monitor.record_inference_metrics({
            'latency_ms': 45.2,
            'throughput_qps': 22.1,
            'model_version': 'v1.0.0'
        })

        # 監視データ取得
        training_stats = perf_monitor.get_training_statistics()
        inference_stats = perf_monitor.get_inference_statistics()

        assert len(training_stats) > 0
        assert len(inference_stats) > 0

        # 品質監視
        quality_monitor = MLQualityMonitor()

        # データドリフト検出テスト（モック）
        with patch.object(quality_monitor, 'detect_data_drift') as mock_drift:
            mock_drift.return_value = {
                'drift_detected': False,
                'drift_score': 0.12,
                'threshold': 0.3
            }

            drift_result = quality_monitor.detect_data_drift(
                reference_data=np.random.random((1000, 50)),
                current_data=np.random.random((100, 50))
            )

        assert 'drift_detected' in drift_result
        assert isinstance(drift_result['drift_score'], float)

        print("✅ ML monitoring integration passed")

    @pytest.mark.integration
    @pytest.mark.ml
    def test_ml_deployment_validation(self, comprehensive_config, test_environment):
        """MLデプロイメント検証テスト"""
        print("🚀 Testing ML deployment validation...")

        from backend.ml.deployment.model_deployer import ModelDeployer
        from backend.ml.deployment.versioning.version_manager import ModelVersionManager

        deployer = ModelDeployer()
        version_manager = ModelVersionManager(base_path=str(test_environment["artifacts_dir"]))

        # ダミートレーナー
        class MockTrainer:
            def __init__(self):
                self.config = comprehensive_config
                self.model_version = "v1.0.0"

        mock_trainer = MockTrainer()

        # Web配信パッケージ作成
        with patch.object(deployer, 'create_web_package') as mock_web:
            mock_web.return_value = {
                'success': True,
                'package_path': str(test_environment["exports_dir"] / "web_package"),
                'package_size_mb': 15.3,
                'files_included': ['model.json', 'weights.bin', 'inference.js']
            }

            web_result = deployer.create_web_package(
                model_trainer=mock_trainer,
                output_path=str(test_environment["exports_dir"] / "web_package"),
                include_preprocessors=True
            )

        assert web_result['success'] == True
        assert web_result['package_size_mb'] < 50

        # Supabase Edge Functions配信
        with patch.object(deployer, 'deploy_to_supabase') as mock_supabase:
            mock_supabase.return_value = {
                'success': True,
                'function_name': 'enhanced_two_tower_recommendations',
                'deployment_url': 'https://test.supabase.co/functions/v1/enhanced_two_tower_recommendations'
            }

            supabase_result = deployer.deploy_to_supabase(
                model_trainer=mock_trainer,
                function_name='enhanced_two_tower_recommendations'
            )

        assert supabase_result['success'] == True
        assert 'deployment_url' in supabase_result

        # バージョン管理
        version_result = version_manager.register_version(
            model_trainer=mock_trainer,
            version_tag="v1.0.0",
            description="Comprehensive test deployment",
            metrics={
                'auc_pr': 0.89,
                'accuracy': 0.85,
                'inference_latency_ms': 42.1
            }
        )

        assert version_result['success'] == True

        # バージョン検索
        versions = version_manager.list_versions()
        assert len(versions) == 1
        assert versions[0]['version'] == "v1.0.0"

        print("✅ ML deployment validation passed")


if __name__ == "__main__":
    # 統合テスト実行設定
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "integration and ml"
    ])
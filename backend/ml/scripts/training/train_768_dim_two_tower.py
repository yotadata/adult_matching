#!/usr/bin/env python3
"""
768次元Two-Towerモデル訓練スクリプト

既存の評価ベース疑似ユーザーデータを使用して768次元Two-Towerモデルを訓練し、
PostgreSQL pgvectorとの統合に適したモデルを生成する。
"""

import os
import sys
import json
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_pipeline.training.rating_based_two_tower_trainer import RatingBasedTwoTowerTrainer
import tensorflow as tf
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """768次元Two-Towerモデルの訓練実行"""
    
    logger.info("=== 768次元Two-Towerモデル訓練開始 ===")
    
    # 768次元設定でトレーナー初期化（超高速テスト用）
    trainer = RatingBasedTwoTowerTrainer(
        embedding_dim=768,
        learning_rate=0.01,   # 高学習率で超高速化
        batch_size=1024,      # 最大バッチサイズ
        epochs=3,            # 最小限のエポック数
        validation_split=0.2
    )
    
    logger.info("768次元設定完了")
    
    try:
        # データ読み込み
        logger.info("疑似ユーザーデータ読み込み...")
        data_dir = project_root / "data_processing" / "processed_data"
        pseudo_users_file = data_dir / "rating_based_pseudo_users.json"
        integrated_reviews_file = data_dir / "integrated_reviews.json"
        
        if not pseudo_users_file.exists():
            logger.error(f"疑似ユーザーデータが見つかりません: {pseudo_users_file}")
            logger.info("まず 'make generate-rating-users' を実行してください")
            return 1
            
        if not integrated_reviews_file.exists():
            logger.error(f"統合レビューデータが見つかりません: {integrated_reviews_file}")
            logger.info("まず 'make integrate-batch-data' を実行してください") 
            return 1
        
        pseudo_users, reviews = trainer.load_pseudo_user_data(
            str(pseudo_users_file),
            str(integrated_reviews_file)
        )
        
        if not pseudo_users or not reviews:
            logger.error("データ読み込みに失敗しました")
            return 1
            
        # 訓練データ準備
        logger.info("768次元モデル用訓練データ準備...")
        content_df, interactions_df = trainer.prepare_training_data(pseudo_users, reviews)
        
        # ステップ実行（パス問題を回避）
        logger.info("768次元Two-Towerモデル訓練ステップ実行...")
        
        # Step 3: 特徴量エンジニアリング
        feature_info, user_features, item_features, interactions_df = trainer.prepare_features(content_df, interactions_df)
        
        # Step 4: モデル構築
        model = trainer.build_two_tower_model(
            feature_info['user_vocab_size'],
            feature_info['item_vocab_size'], 
            feature_info['user_feature_dim'],
            feature_info['item_feature_dim']
        )
        
        # Step 5: データセット作成
        train_dataset, val_dataset = trainer.create_training_dataset(
            interactions_df, user_features, item_features, feature_info
        )
        
        # Step 6: 訓練実行
        history = trainer.train_model(model, train_dataset, val_dataset)
        
        # Step 7: 評価
        metrics = trainer.evaluate_model(model, val_dataset)
        
        if not metrics:
            logger.error("モデル訓練に失敗しました")
            return 1
            
        # 768次元モデル専用ディレクトリに直接保存
        output_dir = project_root / "ml_pipeline" / "models" / "rating_based_two_tower_768"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("768次元モデルを専用ディレクトリに保存...")
        
        # Step 8: 768次元専用ディレクトリに直接保存
        model.save(output_dir / "full_model_768.keras")
        trainer.user_tower.save(output_dir / "user_tower_768.keras")
        trainer.item_tower.save(output_dir / "item_tower_768.keras")
        
        # 前処理器・メタデータ保存
        import pickle
        import json
        
        with open(output_dir / "preprocessors_768.pkl", 'wb') as f:
            pickle.dump({
                'text_vectorizer': trainer.text_vectorizer,
                'category_encoder': trainer.category_encoder,
                'scaler': trainer.scaler
            }, f)
            
        with open(output_dir / "feature_info_768.json", 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "training_stats_768.json", 'w', encoding='utf-8') as f:
            json.dump(trainer.training_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"768次元Two-Towerモデル保存完了: {output_dir}")
        for file_name in ["full_model_768.keras", "user_tower_768.keras", "item_tower_768.keras", 
                         "preprocessors_768.pkl", "feature_info_768.json", "training_stats_768.json"]:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"✓ 保存確認: {file_path}")
            else:
                logger.error(f"✗ 保存失敗: {file_path}")
        
        # 訓練統計確認
        stats = trainer.training_stats
        evaluation = stats.get('evaluation_metrics', {})
        
        logger.info("=== 768次元モデル訓練結果 ===")
        logger.info(f"AUC-ROC: {evaluation.get('auc_roc', 'N/A')}")
        logger.info(f"AUC-PR: {evaluation.get('auc_pr', 'N/A')}")  
        logger.info(f"Accuracy: {evaluation.get('accuracy', 'N/A')}")
        logger.info(f"埋め込み次元: 768次元")
        logger.info(f"ユーザー数: {stats['data_stats'].get('total_users', 'N/A')}")
        logger.info(f"アイテム数: {stats['data_stats'].get('total_items', 'N/A')}")
        logger.info(f"インタラクション数: {stats['data_stats'].get('total_interactions', 'N/A')}")
        
        logger.info("=== 768次元Two-Towerモデル訓練完了 ===")
        return 0
        
    except Exception as e:
        logger.error(f"訓練中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
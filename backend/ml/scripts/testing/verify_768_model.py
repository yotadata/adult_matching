#!/usr/bin/env python3
"""
768次元Two-Towerモデル検証スクリプト

保存された768次元モデルが正しく768次元出力を生成するかテストする。
"""

import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_768_model():
    """768次元モデルの検証"""
    
    logger.info("=== 768次元Two-Towerモデル検証開始 ===")
    
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "ml_pipeline" / "models" / "rating_based_two_tower_768"
    
    try:
        # ユーザータワーモデル検証
        user_tower_path = model_dir / "user_tower_768.keras"
        if user_tower_path.exists():
            logger.info("User Towerモデル検証...")
            user_model = tf.keras.models.load_model(user_tower_path, safe_mode=False)
            
            # モデル構造確認
            logger.info("User Towerモデル構造:")
            user_model.summary(print_fn=logger.info)
            
            # テスト入力でベクトル次元確認
            test_user_id = np.array([0])  # ユーザーID
            test_user_features = np.random.randn(1, 3).astype(np.float32)  # 3次元特徴量
            
            user_embedding = user_model.predict([test_user_id, test_user_features])
            logger.info(f"User Tower出力形状: {user_embedding.shape}")
            logger.info(f"User Tower出力次元: {user_embedding.shape[1]}次元")
            
            # L2正規化確認
            norm = np.linalg.norm(user_embedding[0])
            logger.info(f"User Tower出力L2ノルム: {norm:.6f}")
            
        else:
            logger.error(f"User Towerモデルが見つかりません: {user_tower_path}")
            return False
            
        # アイテムタワーモデル検証
        item_tower_path = model_dir / "item_tower_768.keras"
        if item_tower_path.exists():
            logger.info("Item Towerモデル検証...")
            item_model = tf.keras.models.load_model(item_tower_path, safe_mode=False)
            
            # モデル構造確認
            logger.info("Item Towerモデル構造:")
            item_model.summary(print_fn=logger.info)
            
            # テスト入力でベクトル次元確認
            test_item_id = np.array([0])  # アイテムID
            test_item_features = np.random.randn(1, 1003).astype(np.float32)  # 1003次元特徴量
            
            item_embedding = item_model.predict([test_item_id, test_item_features])
            logger.info(f"Item Tower出力形状: {item_embedding.shape}")
            logger.info(f"Item Tower出力次元: {item_embedding.shape[1]}次元")
            
            # L2正規化確認
            norm = np.linalg.norm(item_embedding[0])
            logger.info(f"Item Tower出力L2ノルム: {norm:.6f}")
            
        else:
            logger.error(f"Item Towerモデルが見つかりません: {item_tower_path}")
            return False
            
        # コサイン類似度計算テスト
        logger.info("コサイン類似度計算テスト...")
        cosine_similarity = np.dot(user_embedding[0], item_embedding[0])
        logger.info(f"User-Itemコサイン類似度: {cosine_similarity:.6f}")
        
        # フルモデル検証
        full_model_path = model_dir / "full_model_768.keras"
        if full_model_path.exists():
            logger.info("フルモデル検証...")
            full_model = tf.keras.models.load_model(full_model_path, safe_mode=False)
            
            prediction = full_model.predict([test_user_id, test_user_features, test_item_id, test_item_features])
            logger.info(f"フルモデル予測: {prediction[0][0]:.6f}")
            
        else:
            logger.error(f"フルモデルが見つかりません: {full_model_path}")
            return False
        
        # 検証結果判定
        user_is_768 = user_embedding.shape[1] == 768
        item_is_768 = item_embedding.shape[1] == 768
        user_normalized = abs(norm - 1.0) < 0.01
        item_norm = np.linalg.norm(item_embedding[0])
        item_normalized = abs(item_norm - 1.0) < 0.01
        
        logger.info("=== 検証結果 ===")
        logger.info(f"✅ User Tower 768次元: {'○' if user_is_768 else '✗'}")
        logger.info(f"✅ Item Tower 768次元: {'○' if item_is_768 else '✗'}")
        logger.info(f"✅ User Tower L2正規化: {'○' if user_normalized else '✗'}")
        logger.info(f"✅ Item Tower L2正規化: {'○' if item_normalized else '✗'}")
        
        success = user_is_768 and item_is_768 and user_normalized and item_normalized
        
        if success:
            logger.info("🎉 768次元Two-Towerモデル検証成功！")
            logger.info("PostgreSQL pgvector(768)との統合準備完了")
            return True
        else:
            logger.error("❌ 768次元モデル検証失敗")
            return False
            
    except Exception as e:
        logger.error(f"検証中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = verify_768_model()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
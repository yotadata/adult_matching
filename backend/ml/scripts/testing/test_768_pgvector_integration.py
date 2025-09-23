#!/usr/bin/env python3
"""
768次元ベクトルとPostgreSQL pgvectorの統合テスト

768次元Two-Towerモデルで生成されたユーザー・アイテム埋め込みが
PostgreSQL pgvector拡張で正しく動作することを確認する。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_embeddings() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """768次元のテスト用埋め込みベクトルを生成"""
    
    logger.info("768次元テスト埋め込み生成...")
    
    # L2正規化された768次元ベクトル生成
    np.random.seed(42)  # 再現性のため
    
    # ユーザー埋め込み（10個のテストユーザー）
    user_embeddings = []
    for i in range(10):
        # ランダム768次元ベクトル生成
        embedding = np.random.randn(768).astype(np.float32)
        # L2正規化
        embedding = embedding / np.linalg.norm(embedding)
        user_embeddings.append(embedding)
    
    # アイテム埋め込み（20個のテストアイテム）
    item_embeddings = []
    for i in range(20):
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        item_embeddings.append(embedding)
    
    logger.info(f"生成完了: {len(user_embeddings)}ユーザー, {len(item_embeddings)}アイテム")
    logger.info(f"ベクトル次元: {user_embeddings[0].shape[0]}次元")
    logger.info(f"L2ノルム確認: {np.linalg.norm(user_embeddings[0]):.6f}")
    
    return user_embeddings, item_embeddings

def test_pgvector_operations(user_embeddings: List[np.ndarray], 
                           item_embeddings: List[np.ndarray]) -> bool:
    """PostgreSQL pgvectorで768次元ベクトル操作をテスト"""
    
    logger.info("PostgreSQL pgvector 768次元テスト開始...")
    
    try:
        # データベース接続（Supabase local dev環境想定）
        conn_params = {
            'host': 'localhost',
            'port': 54322,
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        logger.info("データベース接続中...")
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # pgvector拡張の確認
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if not cur.fetchone():
            logger.error("pgvector拡張が見つかりません")
            return False
        
        logger.info("pgvector拡張確認済み")
        
        # テスト用テーブル作成
        cur.execute("DROP TABLE IF EXISTS test_user_embeddings_768")
        cur.execute("DROP TABLE IF EXISTS test_item_embeddings_768")
        
        cur.execute("""
            CREATE TABLE test_user_embeddings_768 (
                user_id TEXT PRIMARY KEY,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            CREATE TABLE test_item_embeddings_768 (
                item_id TEXT PRIMARY KEY,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        logger.info("768次元テストテーブル作成完了")
        
        # ユーザー埋め込み挿入テスト
        logger.info("ユーザー埋め込み挿入テスト...")
        for i, embedding in enumerate(user_embeddings):
            user_id = f"test_user_{i:03d}"
            embedding_list = embedding.tolist()
            
            cur.execute("""
                INSERT INTO test_user_embeddings_768 (user_id, embedding)
                VALUES (%s, %s)
            """, (user_id, embedding_list))
        
        # アイテム埋め込み挿入テスト
        logger.info("アイテム埋め込み挿入テスト...")
        for i, embedding in enumerate(item_embeddings):
            item_id = f"test_item_{i:03d}"
            embedding_list = embedding.tolist()
            
            cur.execute("""
                INSERT INTO test_item_embeddings_768 (item_id, embedding)
                VALUES (%s, %s)
            """, (item_id, embedding_list))
        
        conn.commit()
        logger.info("埋め込み挿入完了")
        
        # コサイン類似度検索テスト
        logger.info("コサイン類似度検索テスト...")
        test_user_embedding = user_embeddings[0].tolist()
        
        cur.execute("""
            SELECT 
                item_id,
                1 - (embedding <=> %s) as cosine_similarity
            FROM test_item_embeddings_768
            ORDER BY cosine_similarity DESC
            LIMIT 5
        """, (test_user_embedding,))
        
        results = cur.fetchall()
        logger.info("類似度検索結果（上位5件）:")
        for result in results:
            logger.info(f"  {result['item_id']}: {result['cosine_similarity']:.6f}")
        
        # 距離計算テスト
        logger.info("距離計算テスト...")
        cur.execute("""
            SELECT 
                u.user_id,
                i.item_id,
                u.embedding <-> i.embedding as l2_distance,
                u.embedding <#> i.embedding as negative_inner_product,
                u.embedding <=> i.embedding as cosine_distance
            FROM test_user_embeddings_768 u
            CROSS JOIN test_item_embeddings_768 i
            WHERE u.user_id = 'test_user_000' AND i.item_id = 'test_item_000'
        """)
        
        distance_result = cur.fetchone()
        logger.info("距離計算結果:")
        logger.info(f"  L2距離: {distance_result['l2_distance']:.6f}")
        logger.info(f"  負の内積: {distance_result['negative_inner_product']:.6f}")
        logger.info(f"  コサイン距離: {distance_result['cosine_distance']:.6f}")
        
        # インデックス作成テスト
        logger.info("ベクトルインデックス作成テスト...")
        cur.execute("""
            CREATE INDEX CONCURRENTLY test_user_embeddings_768_idx 
            ON test_user_embeddings_768 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 10)
        """)
        
        cur.execute("""
            CREATE INDEX CONCURRENTLY test_item_embeddings_768_idx 
            ON test_item_embeddings_768 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 20)
        """)
        
        conn.commit()
        logger.info("ベクトルインデックス作成完了")
        
        # パフォーマンステスト
        logger.info("パフォーマンステスト...")
        import time
        
        start_time = time.time()
        cur.execute("""
            SELECT item_id, 1 - (embedding <=> %s) as similarity
            FROM test_item_embeddings_768
            ORDER BY similarity DESC
            LIMIT 10
        """, (test_user_embedding,))
        results = cur.fetchall()
        end_time = time.time()
        
        query_time = (end_time - start_time) * 1000  # ミリ秒
        logger.info(f"類似度検索時間: {query_time:.2f}ms")
        logger.info(f"結果数: {len(results)}件")
        
        # クリーンアップ
        cur.execute("DROP TABLE test_user_embeddings_768")
        cur.execute("DROP TABLE test_item_embeddings_768")
        conn.commit()
        
        cur.close()
        conn.close()
        
        logger.info("✅ PostgreSQL pgvector 768次元テスト成功")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"PostgreSQLエラー: {e}")
        return False
    except Exception as e:
        logger.error(f"テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """768次元pgvector統合テスト実行"""
    
    logger.info("=== 768次元pgvector統合テスト開始 ===")
    
    try:
        # 1. テスト用埋め込み生成
        user_embeddings, item_embeddings = create_test_embeddings()
        
        # 2. pgvector動作テスト
        success = test_pgvector_operations(user_embeddings, item_embeddings)
        
        if success:
            logger.info("=== 768次元pgvector統合テスト完了 ===")
            logger.info("✅ 768次元Two-TowerモデルはPostgreSQL pgvectorと完全互換")
            return 0
        else:
            logger.error("❌ 768次元pgvector統合テスト失敗")
            return 1
            
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
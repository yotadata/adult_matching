-- pgvector 768次元最適化マイグレーション
-- パフォーマンス向上のための設定とインデックス最適化

-- pgvector拡張が有効であることを確認
CREATE EXTENSION IF NOT EXISTS vector;

-- 768次元ベクター用の最適化設定
-- これらの設定はconfig.tomlの設定と連携して動作

-- ベクターインデックスの最適化設定
-- IVFFlat インデックスの最適化（768次元用）
-- 既存のベクターカラムにインデックスが存在する場合の最適化

-- user_embeddingsテーブルの768次元インデックス最適化
DO $$
BEGIN
    -- 既存のインデックスをドロップして再作成（より効率的なパラメータで）
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'user_embeddings_embedding_idx') THEN
        DROP INDEX user_embeddings_embedding_idx;
    END IF;
    
    -- 768次元に最適化されたIVFFlatインデックスを作成
    -- リスト数は sqrt(行数) を基準に設定（768次元用に調整）
    CREATE INDEX user_embeddings_embedding_idx ON user_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);  -- 768次元・大規模データ用最適化
END $$;

-- video_embeddingsテーブルの768次元インデックス最適化
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'video_embeddings_embedding_idx') THEN
        DROP INDEX video_embeddings_embedding_idx;
    END IF;
    
    -- ビデオ埋め込み用のインデックス（大量データ対応）
    CREATE INDEX video_embeddings_embedding_idx ON video_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 500);  -- より大量のビデオデータ用
END $$;

-- パフォーマンス監視用のビューを作成
CREATE OR REPLACE VIEW vector_performance_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%embedding%'
ORDER BY idx_scan DESC;

-- ベクター検索パフォーマンス分析用関数
CREATE OR REPLACE FUNCTION analyze_vector_performance()
RETURNS TABLE(
    table_name text,
    index_name text,
    scans bigint,
    efficiency numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.tablename::text,
        i.indexname::text,
        i.idx_scan,
        CASE 
            WHEN i.idx_scan > 0 THEN 
                ROUND((i.idx_tup_fetch::numeric / i.idx_scan), 2)
            ELSE 0 
        END as efficiency
    FROM pg_stat_user_indexes i
    WHERE i.indexname LIKE '%embedding%'
    ORDER BY i.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- ベクター検索の統計情報をリセットする関数
CREATE OR REPLACE FUNCTION reset_vector_stats()
RETURNS void AS $$
BEGIN
    -- 統計情報をリセット
    SELECT pg_stat_reset();
    RAISE NOTICE 'Vector performance statistics have been reset';
END;
$$ LANGUAGE plpgsql;

-- コメント追加
COMMENT ON VIEW vector_performance_stats IS 'ベクター検索のパフォーマンス統計を表示するビュー';
COMMENT ON FUNCTION analyze_vector_performance() IS 'ベクター検索インデックスの効率性を分析する関数';
COMMENT ON FUNCTION reset_vector_stats() IS 'ベクター検索の統計情報をリセットする関数';

-- パフォーマンス最適化の確認
SELECT 'pgvector 768次元最適化完了' as status;
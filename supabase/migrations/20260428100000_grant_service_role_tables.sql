-- service_role に public スキーマの全テーブルへのアクセスを付与する
-- Supabase マネージド環境では自動付与されるが、セルフホスト環境では明示的に必要

-- RLS をバイパスできるよう BYPASSRLS 属性を付与
ALTER ROLE service_role BYPASSRLS;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- 今後追加されるオブジェクトにも自動付与
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT EXECUTE ON FUNCTIONS TO service_role;

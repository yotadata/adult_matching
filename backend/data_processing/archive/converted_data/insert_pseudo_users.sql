-- レビューデータから生成された疑似ユーザーデータ
-- 注意: 実際のauth.usersテーブルに挿入するには認証システムとの統合が必要

-- profiles テーブル用（auth.usersが必要）
/*
INSERT INTO public.profiles (user_id, display_name, created_at) VALUES
  ('335bc684-42ee-4867-b82a-766e57def0d0', 'レビュワー_81979', '2025-09-27T10:14:20.384718');
INSERT INTO public.profiles (user_id, display_name, created_at) VALUES
  ('23870a24-6fba-4e42-a5d7-371ac572991b', 'レビュワー_293468', '2025-09-27T10:14:20.386000');
INSERT INTO public.profiles (user_id, display_name, created_at) VALUES
  ('0900d8c5-8098-46e1-9afe-92c0753bb82a', 'レビュワー_16769', '2025-09-27T10:14:20.386060');
INSERT INTO public.profiles (user_id, display_name, created_at) VALUES
  ('edda14ef-60de-4905-b40e-da7bc788cb6b', 'レビュワー_199971', '2025-09-27T10:14:20.386239');
INSERT INTO public.profiles (user_id, display_name, created_at) VALUES
  ('e71c69ce-e59a-486b-908d-396fe4551f5c', 'レビュワー_304408', '2025-09-27T10:14:20.386353');
*/

-- user_video_decisions テーブル用
INSERT INTO public.user_video_decisions (user_id, video_id, decision_type, created_at) VALUES
;

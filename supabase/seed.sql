-- Test data for development
-- Insert test genres and tags
INSERT INTO public.tag_groups (id, name) VALUES 
  ('550e8400-e29b-41d4-a716-446655440001', 'ジャンル'),
  ('550e8400-e29b-41d4-a716-446655440002', 'プレイ');

INSERT INTO public.tags (id, name, tag_group_id) VALUES 
  ('550e8400-e29b-41d4-a716-446655440010', 'ドラマ', '550e8400-e29b-41d4-a716-446655440001'),
  ('550e8400-e29b-41d4-a716-446655440011', 'コメディ', '550e8400-e29b-41d4-a716-446655440001'),
  ('550e8400-e29b-41d4-a716-446655440012', 'アクション', '550e8400-e29b-41d4-a716-446655440001'),
  ('550e8400-e29b-41d4-a716-446655440020', 'ソフト', '550e8400-e29b-41d4-a716-446655440002'),
  ('550e8400-e29b-41d4-a716-446655440021', 'ハード', '550e8400-e29b-41d4-a716-446655440002');

-- Insert test performers
INSERT INTO public.performers (id, name) VALUES 
  ('550e8400-e29b-41d4-a716-446655440100', '田中美咲'),
  ('550e8400-e29b-41d4-a716-446655440101', '佐藤花音'),
  ('550e8400-e29b-41d4-a716-446655440102', '鈴木あい');

-- Insert test videos
INSERT INTO public.videos (id, external_id, title, description, duration_seconds, thumbnail_url, preview_video_url, maker, genre, price, sample_video_url, image_urls, source, published_at) VALUES 
  ('550e8400-e29b-41d4-a716-446655441000', 'TEST001', 'テスト動画1', 'これはテスト用の動画です', 3600, 'https://example.com/thumb1.jpg', 'https://example.com/preview1.mp4', 'テストメーカーA', 'ドラマ', 2980, 'https://example.com/sample1.mp4', ARRAY['https://example.com/img1.jpg'], 'test', NOW()),
  ('550e8400-e29b-41d4-a716-446655441001', 'TEST002', 'テスト動画2', 'これもテスト用の動画です', 2700, 'https://example.com/thumb2.jpg', 'https://example.com/preview2.mp4', 'テストメーカーB', 'コメディ', 1980, 'https://example.com/sample2.mp4', ARRAY['https://example.com/img2.jpg'], 'test', NOW()),
  ('550e8400-e29b-41d4-a716-446655441002', 'TEST003', 'テスト動画3', '3番目のテスト動画', 4200, 'https://example.com/thumb3.jpg', 'https://example.com/preview3.mp4', 'テストメーカーC', 'アクション', 3980, 'https://example.com/sample3.mp4', ARRAY['https://example.com/img3.jpg'], 'test', NOW()),
  ('550e8400-e29b-41d4-a716-446655441003', 'TEST004', 'テスト動画4', '4番目のテスト動画', 3300, 'https://example.com/thumb4.jpg', 'https://example.com/preview4.mp4', 'テストメーカーA', 'ドラマ', 2480, 'https://example.com/sample4.mp4', ARRAY['https://example.com/img4.jpg'], 'test', NOW()),
  ('550e8400-e29b-41d4-a716-446655441004', 'TEST005', 'テスト動画5', '5番目のテスト動画', 2900, 'https://example.com/thumb5.jpg', 'https://example.com/preview5.mp4', 'テストメーカーB', 'コメディ', 1780, 'https://example.com/sample5.mp4', ARRAY['https://example.com/img5.jpg'], 'test', NOW());

-- Insert video-tag relationships
INSERT INTO public.video_tags (video_id, tag_id) VALUES 
  ('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440010'),
  ('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440020'),
  ('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440011'),
  ('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440020'),
  ('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440012'),
  ('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440021'),
  ('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440010'),
  ('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440020'),
  ('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440011'),
  ('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440021');

-- Insert video-performer relationships
INSERT INTO public.video_performers (video_id, performer_id) VALUES 
  ('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440100'),
  ('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440101'),
  ('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440102'),
  ('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440100'),
  ('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440101');
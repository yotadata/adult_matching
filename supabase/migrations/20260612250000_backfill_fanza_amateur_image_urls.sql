-- FANZA_AMATEUR の image_urls が null のレコードを thumbnail_url から生成して補完する。
-- thumbnail_url パターン: https://pics.dmm.co.jp/digital/amateur/<code>/<code>jm.jpg
-- → js-001〜js-005 の配列を image_urls に設定する。
UPDATE public.videos
SET image_urls = ARRAY[
  regexp_replace(thumbnail_url, 'jm\.jpg$', 'js-001.jpg'),
  regexp_replace(thumbnail_url, 'jm\.jpg$', 'js-002.jpg'),
  regexp_replace(thumbnail_url, 'jm\.jpg$', 'js-003.jpg'),
  regexp_replace(thumbnail_url, 'jm\.jpg$', 'js-004.jpg'),
  regexp_replace(thumbnail_url, 'jm\.jpg$', 'js-005.jpg')
]
WHERE source = 'FANZA_AMATEUR'
  AND image_urls IS NULL
  AND thumbnail_url LIKE '%jm.jpg';

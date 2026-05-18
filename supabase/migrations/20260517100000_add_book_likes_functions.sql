-- 気になりリスト取得RPC（漫画版）
create or replace function public.get_user_book_likes(
  p_limit int default 40,
  p_offset int default 0
)
returns table (
  id uuid,
  external_id text,
  title text,
  description text,
  thumbnail_url text,
  sample_image_urls text[],
  author text,
  product_url text,
  affiliate_url text,
  price numeric,
  page_count int,
  product_released_at timestamptz,
  tags jsonb,
  liked_at timestamptz
)
language sql stable security definer
as $$
  select
    b.id,
    b.external_id,
    b.title,
    b.description,
    b.thumbnail_url,
    b.sample_image_urls,
    b.author,
    b.product_url,
    b.affiliate_url,
    b.price,
    b.page_count,
    b.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name))
        from public.book_tags bt
        join public.tags t on t.id = bt.tag_id
        where bt.book_id = b.id
      ),
      '[]'::jsonb
    ) as tags,
    bl.created_at as liked_at
  from public.book_likes bl
  join public.books b on b.id = bl.book_id
  where bl.user_id = auth.uid()
  order by bl.created_at desc
  limit p_limit
  offset p_offset;
$$;

grant execute on function public.get_user_book_likes to authenticated;

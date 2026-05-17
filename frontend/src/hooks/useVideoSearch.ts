import { useCallback, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type VideoSearchItem = {
  id: string;
  title: string;
  thumbnail_url: string | null;
  product_url: string | null;
  affiliate_url: string | null;
  product_released_at: string | null;
  author: string | null;
  tags: Array<{ id: string; name: string }>;
};

type UseVideoSearchOptions = {
  keyword: string;
  tagIds?: string[];
  limit?: number;
};

export function useVideoSearch({ keyword, tagIds, limit = 24 }: UseVideoSearchOptions) {
  const [results, setResults] = useState<VideoSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 配列をキー文字列に変換して参照安定化（配列をdepsに入れると毎render変わる）
  const tagIdsKey = (tagIds ?? []).join(',');

  const search = useCallback(async () => {
    const tagIdsArr = tagIdsKey ? tagIdsKey.split(',') : [];
    const trimmed = keyword.trim();
    if (trimmed.length === 0) {
      setResults([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // タグでブックIDを絞り込む
      let filteredIds: Set<string> | null = null;

      if (tagIdsArr.length > 0) {
        const { data: tagRows } = await supabase
          .from('book_tags')
          .select('book_id')
          .in('tag_id', tagIdsArr);
        const ids = new Set((tagRows ?? []).map((r) => r.book_id as string));
        filteredIds = ids;
      }

      // フィルター後に0件なら早期終了
      if (filteredIds !== null && filteredIds.size === 0) {
        setResults([]);
        setLoading(false);
        return;
      }

      let query = supabase
        .from('books')
        .select('id, title, thumbnail_url, product_url, affiliate_url, product_released_at, author')
        .ilike('title', `%${trimmed}%`)
        .order('product_released_at', { ascending: false, nullsFirst: false })
        .limit(limit);

      if (filteredIds !== null) {
        query = query.in('id', [...filteredIds]);
      }

      const { data: books, error: dbError } = await query;
      if (dbError) throw dbError;

      if (!books || books.length === 0) {
        setResults([]);
        return;
      }

      const ids = books.map((v) => v.id);

      const { data: tagData } = await supabase
        .from('book_tags')
        .select('book_id, tags(id, name)')
        .in('book_id', ids);

      setResults(
        books.map((v) => ({
          ...v,
          tags: (tagData ?? [])
            .filter((t) => t.book_id === v.id)
            .flatMap((t) => {
              const tag = t.tags as unknown as { id: string; name: string } | null;
              return tag ? [tag] : [];
            }),
        })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : '検索に失敗しました');
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyword, tagIdsKey, limit]);

  useEffect(() => {
    search();
  }, [search]);

  return { results, loading, error };
}

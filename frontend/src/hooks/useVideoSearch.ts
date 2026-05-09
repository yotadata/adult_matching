import { useCallback, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type VideoSearchItem = {
  id: string;
  title: string;
  thumbnail_url: string | null;
  product_url: string | null;
  affiliate_url: string | null;
  sample_video_url: string | null;
  product_released_at: string | null;
  tags: Array<{ id: string; name: string }>;
  performers: Array<{ id: string; name: string }>;
};

type UseVideoSearchOptions = {
  keyword: string;
  tagIds?: string[];
  performerIds?: string[];
  limit?: number;
};

export function useVideoSearch({ keyword, tagIds = [], performerIds = [], limit = 24 }: UseVideoSearchOptions) {
  const [results, setResults] = useState<VideoSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async () => {
    const trimmed = keyword.trim();
    if (trimmed.length === 0) {
      setResults([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // タグ・出演者でビデオIDを絞り込む
      let filteredIds: Set<string> | null = null;

      if (tagIds.length > 0) {
        const { data: tagRows } = await supabase
          .from('video_tags')
          .select('video_id')
          .in('tag_id', tagIds);
        const ids = new Set((tagRows ?? []).map((r) => r.video_id as string));
        filteredIds = ids;
      }

      if (performerIds.length > 0) {
        const { data: perfRows } = await supabase
          .from('video_performers')
          .select('video_id')
          .in('performer_id', performerIds);
        const ids = new Set((perfRows ?? []).map((r) => r.video_id as string));
        filteredIds = filteredIds
          ? new Set([...filteredIds].filter((id) => ids.has(id)))
          : ids;
      }

      // フィルター後に0件なら早期終了
      if (filteredIds !== null && filteredIds.size === 0) {
        setResults([]);
        setLoading(false);
        return;
      }

      let query = supabase
        .from('videos')
        .select('id, title, thumbnail_url, product_url, affiliate_url, sample_video_url, product_released_at')
        .ilike('title', `%${trimmed}%`)
        .order('product_released_at', { ascending: false, nullsFirst: false })
        .limit(limit);

      if (filteredIds !== null) {
        query = query.in('id', [...filteredIds]);
      }

      const { data: videos, error: dbError } = await query;
      if (dbError) throw dbError;

      if (!videos || videos.length === 0) {
        setResults([]);
        return;
      }

      const ids = videos.map((v) => v.id);

      const [{ data: tagData }, { data: perfData }] = await Promise.all([
        supabase.from('video_tags').select('video_id, tags(id, name)').in('video_id', ids),
        supabase.from('video_performers').select('video_id, performers(id, name)').in('video_id', ids),
      ]);

      setResults(
        videos.map((v) => ({
          ...v,
          tags: (tagData ?? [])
            .filter((t) => t.video_id === v.id)
            .flatMap((t) => {
              const tag = t.tags as unknown as { id: string; name: string } | null;
              return tag ? [tag] : [];
            }),
          performers: (perfData ?? [])
            .filter((p) => p.video_id === v.id)
            .flatMap((p) => {
              const perf = p.performers as unknown as { id: string; name: string } | null;
              return perf ? [perf] : [];
            }),
        })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : '検索に失敗しました');
    } finally {
      setLoading(false);
    }
  }, [keyword, tagIds, performerIds, limit]);

  useEffect(() => {
    search();
  }, [search]);

  return { results, loading, error };
}

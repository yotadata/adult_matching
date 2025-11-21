'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import VideoListDrawer, {
  VideoRecord,
  TagFilterWithGroup,
  PerformerFilterOption,
  SortKey,
  SortOrder,
} from './VideoListDrawer';

type RpcVideoRecord = VideoRecord & { total_count?: number | null };

interface LikedVideosDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>('liked_at');
  const [order, setOrder] = useState<SortOrder>('desc');
  const [tagOptions, setTagOptions] = useState<TagFilterWithGroup[]>([]);
  const [performerOptions, setPerformerOptions] = useState<PerformerFilterOption[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [selectedPerformerIds, setSelectedPerformerIds] = useState<string[]>([]);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const pageSize = 40;
  const [page, setPage] = useState(0);

  const toggleTag = (id: string) => {
    setSelectedTagIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
    setPage(0);
  };

  const togglePerformer = (id: string) => {
    setSelectedPerformerIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
    setPage(0);
  };

  const handleSortChange = (value: SortKey) => {
    setPage(0);
    setSort(value);
  };

  const handleOrderChange = (value: SortOrder) => {
    setPage(0);
    setOrder(value);
  };

  useEffect(() => {
    if (!isOpen) return;

    (async () => {
      try {
        const [{ data: tags }, { data: performers }] = await Promise.all([
          supabase.rpc('get_user_liked_tags'),
          supabase.rpc('get_user_liked_performers'),
        ]);
        setTagOptions((tags as TagFilterWithGroup[] | null)?.filter(Boolean) ?? []);
        setPerformerOptions((performers as PerformerFilterOption[] | null)?.filter(Boolean) ?? []);
      } catch {
        /* noop */
      }
    })();
  }, [isOpen]);

  useEffect(() => {
    const fetchLikedVideos = async () => {
      if (!isOpen) return;
      setLoading(true);
      setError(null);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setVideos([]);
        setTotalCount(null);
        setLoading(false);
        return;
      }

      const { data, error } = await supabase.rpc('get_user_likes', {
        p_search: null,
        p_sort: sort,
        p_order: order,
        p_limit: pageSize,
        p_offset: page * pageSize,
        p_price_min: null,
        p_price_max: null,
        p_release_gte: null,
        p_release_lte: null,
        p_tag_ids: selectedTagIds.length ? selectedTagIds : null,
        p_performer_ids: selectedPerformerIds.length ? selectedPerformerIds : null,
      });
      if (error) {
        setError(error.message);
        setVideos([]);
      } else {
        const rows = (data as RpcVideoRecord[]) ?? [];
        const nextVideos = rows.map((row) => ({
          ...row,
          source: row.source || 'personalized',
        }));
        const nextTotal = rows.length > 0
          ? rows[0]?.total_count ?? rows.length
          : 0;
        setVideos(nextVideos);
        setTotalCount(nextTotal);
      }
      setLoading(false);
    };

    fetchLikedVideos();
  }, [isOpen, sort, order, selectedTagIds, selectedPerformerIds, page]);

  useEffect(() => {
    if (totalCount === null) return;
    const maxPage = Math.max(0, Math.ceil(totalCount / pageSize) - 1);
    if (page > maxPage) {
      setPage(maxPage);
    }
  }, [totalCount, pageSize, page]);

  return (
    <VideoListDrawer
      isOpen={isOpen}
      onClose={() => {
        setSelectedTagIds([]);
        setSelectedPerformerIds([]);
        setPage(0);
        onClose();
      }}
      title="いいねした作品"
      videos={videos}
      loading={loading}
      error={error}
      sort={sort}
      order={order}
      onChangeSort={handleSortChange}
      onChangeOrder={handleOrderChange}
      tagOptions={tagOptions}
      performerOptions={performerOptions}
      selectedTagIds={selectedTagIds}
      selectedPerformerIds={selectedPerformerIds}
      onToggleTag={toggleTag}
      onTogglePerformer={togglePerformer}
      totalCount={totalCount}
      page={page}
      onChangePage={setPage}
      pageSize={pageSize}
    />
  );
};

export default LikedVideosDrawer;

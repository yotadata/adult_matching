'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import VideoListDrawer, {
  VideoRecord,
  TagFilterOption,
  PerformerFilterOption,
  SortKey,
  SortOrder,
} from './VideoListDrawer';

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
  const [tagOptions, setTagOptions] = useState<TagFilterOption[]>([]);
  const [performerOptions, setPerformerOptions] = useState<PerformerFilterOption[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [selectedPerformerIds, setSelectedPerformerIds] = useState<string[]>([]);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const pageSize = 40;

  const toggleTag = (id: string) => {
    setSelectedTagIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
  };

  const togglePerformer = (id: string) => {
    setSelectedPerformerIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
  };

  useEffect(() => {
    if (!isOpen) return;

    (async () => {
      try {
        const [{ data: tags }, { data: performers }] = await Promise.all([
          supabase.rpc('get_user_liked_tags'),
          supabase.rpc('get_user_liked_performers'),
        ]);
        setTagOptions((tags as TagFilterOption[] | null)?.filter(Boolean) ?? []);
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

      const { count } = await supabase
        .from('user_video_decisions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('decision_type', 'like');
      setTotalCount(typeof count === 'number' ? count : null);

      const { data, error } = await supabase.rpc('get_user_likes', {
        p_search: null,
        p_sort: sort,
        p_order: order,
        p_limit: pageSize,
        p_offset: 0,
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
        setVideos(((data as VideoRecord[]) ?? []).map((row) => ({ ...row, source: row.source || 'personalized' })));
      }
      setLoading(false);
    };

    fetchLikedVideos();
  }, [isOpen, sort, order, selectedTagIds, selectedPerformerIds]);

  return (
    <VideoListDrawer
      isOpen={isOpen}
      onClose={() => {
        setSelectedTagIds([]);
        setSelectedPerformerIds([]);
        onClose();
      }}
      title="いいねした作品"
      videos={videos}
      loading={loading}
      error={error}
      sort={sort}
      order={order}
      onChangeSort={setSort}
      onChangeOrder={setOrder}
      tagOptions={tagOptions}
      performerOptions={performerOptions}
      selectedTagIds={selectedTagIds}
      selectedPerformerIds={selectedPerformerIds}
      onToggleTag={toggleTag}
      onTogglePerformer={togglePerformer}
      totalCount={totalCount}
    />
  );
};

export default LikedVideosDrawer;

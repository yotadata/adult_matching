/**
 * 標準クエリパターン - Tags-based Genre System
 *
 * 全videosクエリにtags JOINを標準化するクエリパターンを定義
 * genreカラム参照を完全に除去したクエリテンプレートを提供
 */

import { SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3';

// 基本的なvideos with tagsクエリパターン
export const VIDEOS_WITH_TAGS_SELECT = `
  id,
  title,
  description,
  thumbnail_url,
  maker,
  price,
  distribution_started_at,
  product_released_at,
  sample_video_url,
  image_urls,
  source,
  published_at,
  created_at
`;

// ジャンル取得のサブクエリパターン
export const GENRE_SUBQUERY = `COALESCE((SELECT t.name FROM video_tags vt JOIN tags t ON vt.tag_id = t.id JOIN tag_groups tg ON t.tag_group_id = tg.id WHERE vt.video_id = videos.id AND tg.name ILIKE '%genre%' ORDER BY t.name LIMIT 1), 'Unknown') as genre`;

// 全タグ取得のサブクエリパターン
export const ALL_TAGS_SUBQUERY = `COALESCE((SELECT jsonb_agg(t.name ORDER BY t.name) FROM video_tags vt JOIN tags t ON vt.tag_id = t.id WHERE vt.video_id = videos.id), '[]'::jsonb) as all_tags`;

// 完全なvideos + tags JOINクエリパターン
export const VIDEOS_WITH_TAGS_FULL_SELECT = `
  ${VIDEOS_WITH_TAGS_SELECT},
  ${GENRE_SUBQUERY},
  ${ALL_TAGS_SUBQUERY}
`;

/**
 * 標準的なvideosクエリ（ジャンル付き）を取得
 */
export function getVideosWithTagsQuery(supabaseClient: SupabaseClient) {
  return supabaseClient
    .from('videos')
    .select(VIDEOS_WITH_TAGS_FULL_SELECT);
}

/**
 * 特定のvideoをIDで取得（ジャンル付き）
 */
export function getVideoByIdWithTags(supabaseClient: SupabaseClient, videoId: string) {
  return supabaseClient
    .from('videos')
    .select(VIDEOS_WITH_TAGS_FULL_SELECT)
    .eq('id', videoId)
    .single();
}

/**
 * ジャンルでフィルタリングされたvideosクエリ
 * genreフィルタの代替として、tags名での絞り込みを提供
 */
export function getVideosByGenresWithTags(
  supabaseClient: SupabaseClient,
  genres: string[]
) {
  if (genres.length === 0) {
    return getVideosWithTagsQuery(supabaseClient);
  }

  // tags名でフィルタリング
  return supabaseClient
    .from('videos')
    .select(`
      ${VIDEOS_WITH_TAGS_FULL_SELECT}
    `)
    .in('id',
      supabaseClient
        .from('video_tags')
        .select('video_id')
        .in('tag_id',
          supabaseClient
            .from('tags')
            .select('id')
            .in('name', genres)
        )
    );
}

/**
 * ユーザーのいいね履歴からジャンル嗜好を分析（tags-based）
 */
export function getUserGenrePreferencesQuery(
  supabaseClient: SupabaseClient,
  userId: string
) {
  return supabaseClient
    .from('likes')
    .select(`
      videos!inner(
        id,
        ${GENRE_SUBQUERY}
      )
    `)
    .eq('user_id', userId);
}

/**
 * ジャンル多様性計算用のクエリ
 * 動画リストからジャンル分布を取得
 */
export function getGenreDiversityQuery(
  supabaseClient: SupabaseClient,
  videoIds: string[]
) {
  return supabaseClient
    .from('videos')
    .select(`
      id,
      ${GENRE_SUBQUERY}
    `)
    .in('id', videoIds);
}

/**
 * 推薦システム用の高速ジャンル取得
 * パフォーマンス最適化されたクエリ
 */
export function getVideoGenresForRecommendation(
  supabaseClient: SupabaseClient,
  videoIds: string[]
) {
  if (videoIds.length === 0) return Promise.resolve([]);

  return supabaseClient
    .from('videos')
    .select(`
      id,
      ${GENRE_SUBQUERY}
    `)
    .in('id', videoIds)
    .limit(videoIds.length);
}

/**
 * 検索機能用のジャンル検索クエリ
 * テキスト検索にジャンル（tags）検索を統合
 */
export function searchVideosWithGenreQuery(
  supabaseClient: SupabaseClient,
  searchQuery: string,
  genreFilter?: string[]
) {
  let query = supabaseClient
    .from('videos')
    .select(VIDEOS_WITH_TAGS_FULL_SELECT);

  // テキスト検索
  if (searchQuery) {
    query = query.or(`title.ilike.%${searchQuery}%,description.ilike.%${searchQuery}%,maker.ilike.%${searchQuery}%`);
  }

  // ジャンル（tags）フィルタ
  if (genreFilter && genreFilter.length > 0) {
    query = query.in('id',
      supabaseClient
        .from('video_tags')
        .select('video_id')
        .in('tag_id',
          supabaseClient
            .from('tags')
            .select('id')
            .in('name', genreFilter)
        )
    );
  }

  return query;
}

/**
 * エラーハンドリング付きのジャンル取得
 * Unknown ジャンルを保証
 */
export async function getVideoGenreSafe(
  supabaseClient: SupabaseClient,
  videoId: string
): Promise<string> {
  try {
    const { data, error } = await supabaseClient
      .from('videos')
      .select(GENRE_SUBQUERY)
      .eq('id', videoId)
      .single();

    if (error || !data) {
      console.warn(`Failed to get genre for video ${videoId}:`, error);
      return 'Unknown';
    }

    return data.genre || 'Unknown';
  } catch (error) {
    console.error(`Error getting genre for video ${videoId}:`, error);
    return 'Unknown';
  }
}

/**
 * バッチでビデオのジャンルを取得
 * パフォーマンス最適化
 */
export async function getVideoGenresBatch(
  supabaseClient: SupabaseClient,
  videoIds: string[]
): Promise<Record<string, string>> {
  if (videoIds.length === 0) return {};

  try {
    const { data, error } = await supabaseClient
      .from('videos')
      .select(`
        id,
        ${GENRE_SUBQUERY}
      `)
      .in('id', videoIds);

    if (error || !data) {
      console.warn('Failed to get genres for videos:', error);
      // Fallback: すべてのvideoにUnknownを返す
      return videoIds.reduce((acc, id) => {
        acc[id] = 'Unknown';
        return acc;
      }, {} as Record<string, string>);
    }

    // 結果をRecord形式に変換
    const genreMap: Record<string, string> = {};
    data.forEach((item: any) => {
      genreMap[item.id] = item.genre || 'Unknown';
    });

    // 欠落しているvideoIdsにUnknownを設定
    videoIds.forEach(id => {
      if (!(id in genreMap)) {
        genreMap[id] = 'Unknown';
      }
    });

    return genreMap;
  } catch (error) {
    console.error('Error getting genres for videos:', error);
    // Fallback: すべてのvideoにUnknownを返す
    return videoIds.reduce((acc, id) => {
      acc[id] = 'Unknown';
      return acc;
    }, {} as Record<string, string>);
  }
}
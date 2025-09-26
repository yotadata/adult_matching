import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import {
  authenticateUser,
  corsHeaders,
  optionsResponse,
  unauthorizedResponse,
  errorResponse,
  successResponse,
  getSupabaseClient
} from "../../user-management/_shared/auth.ts";
import {
  measureDbOperation
} from "../../user-management/_shared/database.ts";
// Inline type definitions to avoid import issues
interface VideoContent {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
  all_tags: string[];
  created_at?: string;
}

interface ContentFilters {
  genres?: string[];
  makers?: string[];
  min_price?: number;
  max_price?: number;
  performers?: string[];
  tags?: string[];
}

interface PaginationParams {
  limit: number;
  offset: number;
}

// Calculate content metrics inline
function calculateContentMetrics(videos: VideoContent[]) {
  const uniqueGenres = new Set(videos.map(v => v.genre));
  const uniqueMakers = new Set(videos.map(v => v.maker));
  const avgPrice = videos.length > 0 ? videos.reduce((sum, v) => sum + v.price, 0) / videos.length : 0;

  return {
    total_videos: videos.length,
    unique_genres: uniqueGenres.size,
    unique_makers: uniqueMakers.size,
    price_range: {
      min: Math.min(...videos.map(v => v.price)),
      max: Math.max(...videos.map(v => v.price)),
      avg: avgPrice
    }
  };
}

// Search videos with filters inline
async function searchVideosWithFilters(
  supabaseClient: any,
  filters: ContentFilters,
  pagination: PaginationParams
): Promise<{ videos: VideoContent[], total_count: number }> {
  let query = supabaseClient
    .from('videos')
    .select(`
      id,
      title,
      description,
      thumbnail_url,
      preview_video_url,
      maker,
      price,
      sample_video_url,
      image_urls,
      created_at,
      video_performers!inner(performers(name)),
      video_tags!inner(tags(name))
    `);

  // Apply filters
  if (filters.genres && filters.genres.length > 0) {
    query = query.in('id',
      supabaseClient
        .from('video_tags')
        .select('video_id')
        .in('tag_id',
          supabaseClient
            .from('tags')
            .select('id')
            .in('name', filters.genres)
        )
    );
  }

  if (filters.makers && filters.makers.length > 0) {
    query = query.in('maker', filters.makers);
  }

  if (filters.min_price !== undefined) {
    query = query.gte('price', filters.min_price);
  }

  if (filters.max_price !== undefined) {
    query = query.lte('price', filters.max_price);
  }

  const { data: videos, error } = await query
    .range(pagination.offset, pagination.offset + pagination.limit - 1);

  if (error) {
    throw new Error(`Filter search error: ${error.message}`);
  }

  const formattedVideos: VideoContent[] = (videos || []).map((video: any) => {
    const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
    const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

    return {
      id: video.id,
      title: video.title,
      description: video.description,
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      maker: video.maker,
      genre: genreTag,
      price: video.price,
      sample_video_url: video.sample_video_url,
      image_urls: video.image_urls || [],
      performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: tags,
      all_tags: tags,
      created_at: video.created_at
    };
  });

  return {
    videos: formattedVideos,
    total_count: formattedVideos.length
  };
}

// Inline type definition to avoid import issues
interface VideoWithTags {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  duration_seconds?: number;
  genre: string;
  all_tags: string[];
  created_at?: string;
  updated_at?: string;
}

interface SearchRequest {
  query?: string;
  filters?: ContentFilters;
  limit?: number;
  offset?: number;
  sort_by?: 'created_at' | 'price' | 'title' | 'genre';
  sort_order?: 'asc' | 'desc';
}

export default serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  if (req.method !== 'POST') {
    return errorResponse('Method not allowed', 405, 'METHOD_NOT_ALLOWED');
  }

  // ユーザー認証確認（オプショナル）
  const authResult = await authenticateUser(req);
  
  try {
    const supabaseClient = getSupabaseClient();
    
    const {
      query = '',
      filters = {},
      limit = 20,
      offset = 0,
      sort_by = 'created_at',
      sort_order = 'desc'
    }: SearchRequest = await req.json().catch(() => ({}));

    const userId = authResult.authenticated ? authResult.user_id : null;
    console.log(`Searching videos for user: ${userId || 'anonymous'}, query: "${query}"`);

    return await measureDbOperation(async () => {
      const pagination: PaginationParams = { limit, offset };
      
      // テキスト検索の場合
      if (query.trim()) {
        // ジャンルフィルタがある場合は統合クエリを使用
        const genreFilter = filters.genres || [];

        let searchQuery = supabaseClient
          .from('videos')
          .select(`
            id,
            title,
            description,
            thumbnail_url,
            preview_video_url,
            maker,
            price,
            sample_video_url,
            image_urls,
            created_at,
            video_performers!inner(
              performers(name)
            ),
            video_tags!inner(
              tags(name)
            )
          `);

        // テキスト検索
        if (query) {
          searchQuery = searchQuery.or(`title.ilike.%${query}%,description.ilike.%${query}%,maker.ilike.%${query}%`);
        }

        // ジャンル（tags）フィルタ
        if (genreFilter.length > 0) {
          searchQuery = searchQuery.in('id',
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

        const { data: searchResults, error } = await searchQuery
          .order(sort_by, { ascending: sort_order === 'asc' })
          .range(offset, offset + limit - 1);

        if (error) {
          console.error('Text search error:', error);
          throw new Error('Failed to perform text search');
        }

        const formattedResults: VideoContent[] = searchResults?.map((video: any) => {
          const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
          const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

          return {
            id: video.id,
            title: video.title,
            description: video.description,
            thumbnail_url: video.thumbnail_url,
            preview_video_url: video.preview_video_url,
            maker: video.maker,
            genre: genreTag, // tagsから動的に生成されたgenre
            price: video.price,
            sample_video_url: video.sample_video_url,
            image_urls: video.image_urls || [],
            performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
            tags: tags,
            all_tags: tags, // 全タグをall_tagsとして使用
            created_at: video.created_at
          };
        }) || [];

        const metrics = calculateContentMetrics(formattedResults);

        return successResponse({
          videos: formattedResults,
          total_count: searchResults?.length || 0,
          search_type: 'text',
          query,
          filters,
          metrics,
          pagination: {
            limit,
            offset,
            sort_by,
            sort_order,
            has_more: formattedResults.length === limit
          },
          // Tags-based search metadata
          matched_genres: genreFilter.length > 0 ? genreFilter : undefined,
          matched_tags: Array.from(new Set(formattedResults.flatMap(v => v.tags)))
        }, `テキスト検索が完了しました (${formattedResults.length}件)`);
      }
      
      // フィルタ検索（tags-basedフィルタリング）
      else {
        const { videos, total_count } = await searchVideosWithFilters(
          supabaseClient,
          filters,
          pagination
        );

        const metrics = calculateContentMetrics(videos);

        return successResponse({
          videos,
          total_count,
          search_type: 'filter',
          query: '',
          filters,
          metrics,
          pagination: {
            limit,
            offset,
            sort_by,
            sort_order,
            has_more: videos.length === limit
          }
        }, `フィルタ検索が完了しました (${videos.length}件)`);
      }

    }, `searchVideos(query="${query}", user=${userId || 'anon'})`)
      .then(({ result }) => result)
      .catch((error) => {
        console.error('Search error:', error);
        return errorResponse('Failed to search videos', 500, 'SEARCH_ERROR');
      });

  } catch (error) {
    console.error('Unexpected error in search:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
};);
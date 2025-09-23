import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import {
  authenticateUser,
  corsHeaders,
  optionsResponse,
  unauthorizedResponse,
  errorResponse,
  successResponse,
  getSupabaseClient
} from "../_shared/auth.ts";
import {
  checkDuplicateLike,
  measureDbOperation
} from "../_shared/database.ts";

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

interface LikeResponse extends VideoWithTags {
  performers: string[];
  tags: string[];
  liked_at: string;
  purchased: boolean;
}

export default serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  // ユーザー認証確認
  const authResult = await authenticateUser(req);
  if (!authResult.authenticated) {
    return unauthorizedResponse(authResult.error || 'Authentication failed');
  }

  try {
    const supabaseClient = getSupabaseClient();
    const userId = authResult.user_id;

    switch (req.method) {
      case 'GET':
        return await handleGetLikes(supabaseClient, userId, req);
      case 'POST':
        return await handleAddLike(supabaseClient, userId, req);
      case 'DELETE':
        return await handleRemoveLike(supabaseClient, userId, req);
      default:
        return errorResponse('Method not allowed', 405, 'METHOD_NOT_ALLOWED');
    }

  } catch (error) {
    console.error('Unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_SERVER_ERROR');
  }
});

// いいね一覧取得
async function handleGetLikes(supabaseClient: any, userId: string, req: Request) {
  const url = new URL(req.url);
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100);
  const offset = parseInt(url.searchParams.get('offset') || '0');

  return await measureDbOperation(async () => {
    // ユーザーのいいね履歴を取得
    const { data: likes, error } = await supabaseClient
      .from('likes')
      .select(`
        created_at,
        purchased,
        videos!inner(
          id,
          title,
          description,
          thumbnail_url,
          preview_video_url,
          maker,
          price,
          sample_video_url,
          image_urls,
          duration_seconds,
          created_at,
          updated_at,
          video_performers!inner(
            performers(name)
          ),
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Database error:', error);
      throw new Error('Failed to fetch liked videos');
    }

    // ページネーション適用
    const paginatedLikes = likes.slice(offset, offset + limit);

    // データを整形（tags-basedデータ使用）
    const formattedLikes: LikeResponse[] = paginatedLikes.map((like: any) => {
      const tags = like.videos.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

      return {
        id: like.videos.id,
        title: like.videos.title,
        description: like.videos.description,
        thumbnail_url: like.videos.thumbnail_url,
        preview_video_url: like.videos.preview_video_url,
        maker: like.videos.maker,
        genre: genreTag, // tagsから動的に生成されたgenre
        price: like.videos.price,
        sample_video_url: like.videos.sample_video_url,
        image_urls: like.videos.image_urls || [],
        duration_seconds: like.videos.duration_seconds,
        performers: like.videos.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        tags: tags,
        all_tags: tags, // 全タグをall_tagsとして使用
        liked_at: like.created_at,
        purchased: like.purchased || false,
        created_at: like.videos.created_at,
        updated_at: like.videos.updated_at
      };
    });

    return successResponse({
      likes: formattedLikes,
      total_count: likes.length, // 全体のカウント
      pagination: {
        limit,
        offset,
        has_more: (offset + limit) < likes.length
      }
    }, 'いいね一覧を正常に取得しました');

  }, `getUserLikes(${userId}, limit=${limit}, offset=${offset})`)
    .then(({ result }) => result)
    .catch((error) => {
      console.error('Get likes error:', error);
      return errorResponse('Failed to get likes', 500, 'DB_ERROR');
    });
}

// いいね追加
async function handleAddLike(supabaseClient: any, userId: string, req: Request) {
  try {
    const { video_id } = await req.json();

    if (!video_id) {
      return errorResponse('video_id is required', 400, 'MISSING_VIDEO_ID');
    }

    // 重複チェック
    const isDuplicate = await checkDuplicateLike(supabaseClient, userId, video_id);
    if (isDuplicate) {
      return successResponse(
        { message: 'Video already liked', video_id }, 
        'すでにいいね済みです'
      );
    }

    return await measureDbOperation(async () => {
      // 動画が存在するか確認
      const { data: video, error: videoError } = await supabaseClient
        .from('videos')
        .select('id')
        .eq('id', video_id)
        .single();

      if (videoError || !video) {
        throw new Error('Video not found');
      }

      // いいねを追加
      const { error: likeError } = await supabaseClient
        .from('likes')
        .insert({
          user_id: userId,
          video_id: video_id,
          purchased: false,
        });

      if (likeError) {
        console.error('Like insert error:', likeError);
        throw new Error('Failed to add like');
      }

      return successResponse(
        { 
          video_id,
          liked_at: new Date().toISOString()
        }, 
        'いいねを追加しました'
      );

    }, `addLike(${userId}, ${video_id})`)
      .then(({ result }) => result)
      .catch((error) => {
        console.error('Add like error:', error);
        if (error.message === 'Video not found') {
          return errorResponse('Video not found', 404, 'VIDEO_NOT_FOUND');
        }
        return errorResponse('Failed to process like', 500, 'DB_ERROR');
      });

  } catch (error) {
    console.error('Add like error:', error);
    return errorResponse('Failed to process like', 500, 'INTERNAL_ERROR');
  }
}

// いいね削除
async function handleRemoveLike(supabaseClient: any, userId: string, req: Request) {
  try {
    const { video_id } = await req.json();

    if (!video_id) {
      return errorResponse('video_id is required', 400, 'MISSING_VIDEO_ID');
    }

    return await measureDbOperation(async () => {
      // いいねを削除
      const { error: deleteError } = await supabaseClient
        .from('likes')
        .delete()
        .eq('user_id', userId)
        .eq('video_id', video_id);

      if (deleteError) {
        console.error('Like delete error:', deleteError);
        throw new Error('Failed to remove like');
      }

      return successResponse(
        { 
          video_id,
          removed_at: new Date().toISOString()
        }, 
        'いいねを削除しました'
      );

    }, `removeLike(${userId}, ${video_id})`)
      .then(({ result }) => result)
      .catch((error) => {
        console.error('Remove like error:', error);
        return errorResponse('Failed to remove like', 500, 'DB_ERROR');
      });

  } catch (error) {
    console.error('Remove like error:', error);
    return errorResponse('Failed to process unlike', 500, 'INTERNAL_ERROR');
  }
}
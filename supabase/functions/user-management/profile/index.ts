import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { 
  authenticateUser, 
  unauthorizedResponse,
  errorResponse,
  successResponse,
  optionsResponse,
  getSupabaseClientFromRequest
} from "../_shared/auth.ts";
import { 
  measureDbOperation,
  PaginationParams,
  applyPagination
} from "../_shared/database.ts";

interface UserProfile {
  user_id: string;
  email?: string;
  display_name?: string;
  avatar_url?: string;
  bio?: string;
  preferences: {
    content_filters: string[];
    notifications_enabled: boolean;
    privacy_level: 'public' | 'private' | 'friends';
  };
  statistics: {
    total_likes: number;
    account_created: string;
    last_active: string;
    favorite_genres: string[];
    favorite_makers: string[];
  };
}

interface ProfileUpdateRequest {
  display_name?: string;
  bio?: string;
  preferences?: Partial<UserProfile['preferences']>;
}

serve(async (req: Request) => {
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
    const supabaseClient = getSupabaseClientFromRequest(req);
    const userId = authResult.user_id;

    switch (req.method) {
      case 'GET':
        return await handleGetProfile(supabaseClient, userId, req);
      case 'PUT':
        return await handleUpdateProfile(supabaseClient, userId, req);
      case 'DELETE':
        return await handleDeleteProfile(supabaseClient, userId, req);
      default:
        return errorResponse('Method not allowed', 405, 'METHOD_NOT_ALLOWED');
    }

  } catch (error) {
    console.error('Unexpected error:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_SERVER_ERROR');
  }
});

// プロフィール取得
async function handleGetProfile(supabaseClient: any, userId: string, req: Request) {
  const url = new URL(req.url);
  const includeStats = url.searchParams.get('include_stats') === 'true';

  return await measureDbOperation(async () => {
    // ユーザーの基本情報を取得
    const { data: authUser, error: authError } = await supabaseClient.auth.getUser();
    
    if (authError) {
      throw new Error('Failed to get user data');
    }

    // ユーザーのプリファレンスを取得
    const { data: preferences, error: prefError } = await supabaseClient
      .from('user_preferences')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (prefError && prefError.code !== 'PGRST116') { // No rows returned
      console.error('Error getting user preferences:', prefError);
    }

    let statistics = null;
    
    if (includeStats) {
      // 統計情報を取得
      const [likesResult, genreResult, makerResult] = await Promise.all([
        // 総いいね数
        supabaseClient
          .from('likes')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', userId),
        
        // 人気ジャンル
        supabaseClient
          .from('likes')
          .select(`
            videos!inner(
              video_tags!inner(
                tags(name)
              )
            )
          `)
          .eq('user_id', userId)
          .limit(100),
        
        // 人気メーカー
        supabaseClient
          .from('likes')
          .select(`
            videos!inner(maker)
          `)
          .eq('user_id', userId)
          .limit(100)
      ]);

      // ジャンル・メーカー統計の計算
      const genreCount: { [key: string]: number } = {};
      const makerCount: { [key: string]: number } = {};

      if (genreResult.data) {
        genreResult.data.forEach((like: any) => {
          if (like.videos?.video_tags) {
            const tags = like.videos.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);
            const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
            if (genreTag) {
              genreCount[genreTag] = (genreCount[genreTag] || 0) + 1;
            }
          }
        });
      }

      if (makerResult.data) {
        makerResult.data.forEach((like: any) => {
          const maker = like.videos?.maker;
          if (maker) {
            makerCount[maker] = (makerCount[maker] || 0) + 1;
          }
        });
      }

      const favoriteGenres = Object.entries(genreCount)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([genre]) => genre);

      const favoriteMakers = Object.entries(makerCount)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([maker]) => maker);

      statistics = {
        total_likes: likesResult.count || 0,
        account_created: authUser.user.created_at,
        last_active: new Date().toISOString(), // 現在時刻を最終アクティブ時間とする
        favorite_genres: favoriteGenres,
        favorite_makers: favoriteMakers
      };
    }

    const profile: UserProfile = {
      user_id: userId,
      email: authUser.user.email,
      display_name: authUser.user.user_metadata?.display_name || null,
      avatar_url: authUser.user.user_metadata?.avatar_url || null,
      bio: authUser.user.user_metadata?.bio || null,
      preferences: {
        content_filters: preferences?.content_filters || [],
        notifications_enabled: preferences?.notifications_enabled ?? true,
        privacy_level: preferences?.privacy_level || 'private'
      },
      statistics
    };

    return successResponse(profile, 'プロフィールを正常に取得しました');

  }, `getUserProfile(${userId})`)
    .then(({ result }) => result)
    .catch((error) => {
      console.error('Get profile error:', error);
      return errorResponse('Failed to get user profile', 500, 'PROFILE_ERROR');
    });
}

// プロフィール更新
async function handleUpdateProfile(supabaseClient: any, userId: string, req: Request) {
  try {
    const updates: ProfileUpdateRequest = await req.json();

    return await measureDbOperation(async () => {
      const updatePromises = [];

      // ユーザーメタデータの更新
      if (updates.display_name !== undefined || updates.bio !== undefined) {
        const metadataUpdates: any = {};
        if (updates.display_name !== undefined) {
          metadataUpdates.display_name = updates.display_name;
        }
        if (updates.bio !== undefined) {
          metadataUpdates.bio = updates.bio;
        }

        const { error: metadataError } = await supabaseClient.auth.updateUser({
          data: metadataUpdates
        });

        if (metadataError) {
          throw new Error('Failed to update user metadata');
        }
      }

      // プリファレンスの更新
      if (updates.preferences) {
        const { error: prefError } = await supabaseClient
          .from('user_preferences')
          .upsert({
            user_id: userId,
            content_filters: updates.preferences.content_filters,
            notifications_enabled: updates.preferences.notifications_enabled,
            privacy_level: updates.preferences.privacy_level,
            updated_at: new Date().toISOString()
          });

        if (prefError) {
          console.error('Failed to update user preferences:', prefError);
          throw new Error('Failed to update user preferences');
        }
      }

      return successResponse(
        { 
          user_id: userId,
          updated_at: new Date().toISOString(),
          updated_fields: Object.keys(updates)
        }, 
        'プロフィールを正常に更新しました'
      );

    }, `updateUserProfile(${userId})`)
      .then(({ result }) => result)
      .catch((error) => {
        console.error('Update profile error:', error);
        return errorResponse('Failed to update profile', 500, 'UPDATE_ERROR');
      });

  } catch (error) {
    console.error('Update profile error:', error);
    return errorResponse('Invalid request body', 400, 'INVALID_REQUEST');
  }
}

// プロフィール削除（アカウント削除のエイリアス）
async function handleDeleteProfile(supabaseClient: any, userId: string, req: Request) {
  // アカウント削除機能にリダイレクト
  const { default: accountHandler } = await import('../account/index.ts');
  return accountHandler(req);
}
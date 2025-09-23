import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { 
  authenticateUser, 
  corsHeaders, 
  optionsResponse,
  unauthorizedResponse,
  errorResponse,
  successResponse
} from "../_shared/auth.ts";
import { 
  cleanupUserData,
  measureDbOperation
} from "../_shared/database.ts";

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  // Only allow POST method for account deletion
  if (req.method !== 'POST') {
    return errorResponse('Method not allowed', 405, 'METHOD_NOT_ALLOWED');
  }

  // ユーザー認証確認
  const authResult = await authenticateUser(req);
  if (!authResult.authenticated) {
    return unauthorizedResponse(authResult.error || 'Authentication failed');
  }

  try {
    const userId = authResult.user_id;
    console.log(`Starting account deletion process for user: ${userId}`);

    return await measureDbOperation(async () => {
      // Supabase Admin client for privileged operations
      const supabaseAdmin = createClient(
        Deno.env.get('SUPABASE_URL') ?? '',
        Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
        {
          auth: {
            autoRefreshToken: false,
            persistSession: false
          }
        }
      );

      // Use the shared cleanup function for user data
      const cleanupResult = await cleanupUserData(supabaseAdmin, userId);
      
      if (!cleanupResult.success) {
        console.error('User data cleanup failed:', cleanupResult.error);
        throw new Error(cleanupResult.error?.message || 'Failed to cleanup user data');
      }

      // Finally, delete the user from auth.users
      const { error: deleteUserError } = await supabaseAdmin.auth.admin.deleteUser(userId);

      if (deleteUserError) {
        console.error('Failed to delete user from auth:', deleteUserError);
        throw new Error('Failed to delete user account');
      }

      console.log(`Successfully deleted account for user: ${userId}`);

      return successResponse(
        { 
          user_id: userId,
          deleted_at: new Date().toISOString()
        }, 
        'アカウントを正常に削除しました'
      );

    }, `deleteAccount(${userId})`)
      .then(({ result }) => result)
      .catch((error) => {
        console.error('Account deletion failed:', error);
        return errorResponse('Account deletion failed', 500, 'ACCOUNT_DELETION_ERROR');
      });

  } catch (error) {
    console.error('Unexpected error during account deletion:', error);
    return errorResponse('Internal server error', 500, 'INTERNAL_ERROR');
  }
});
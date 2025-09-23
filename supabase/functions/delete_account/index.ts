import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  // Only allow POST method
  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      {
        status: 405,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '', // Service role key for admin operations
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    );

    // ユーザー認証確認
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser();
    if (authError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        {
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const userId = user.id;
    console.log(`Starting account deletion process for user: ${userId}`);

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

    // Begin transaction-like operations
    try {
      // 1. Delete user_embeddings
      const { error: embeddingError } = await supabaseAdmin
        .from('user_embeddings')
        .delete()
        .eq('user_id', userId);

      if (embeddingError) {
        console.error('Failed to delete user embeddings:', embeddingError);
        throw new Error('Failed to delete user embeddings');
      }

      // 2. Delete likes (this will cascade automatically due to foreign key constraints)
      const { error: likesError } = await supabaseAdmin
        .from('likes')
        .delete()
        .eq('user_id', userId);

      if (likesError) {
        console.error('Failed to delete user likes:', likesError);
        throw new Error('Failed to delete user likes');
      }

      // 3. Delete profiles
      const { error: profileError } = await supabaseAdmin
        .from('profiles')
        .delete()
        .eq('user_id', userId);

      if (profileError) {
        console.error('Failed to delete user profile:', profileError);
        throw new Error('Failed to delete user profile');
      }

      // 4. Finally, delete the user from auth.users
      const { error: deleteUserError } = await supabaseAdmin.auth.admin.deleteUser(userId);

      if (deleteUserError) {
        console.error('Failed to delete user from auth:', deleteUserError);
        throw new Error('Failed to delete user account');
      }

      console.log(`Successfully deleted account for user: ${userId}`);

      return new Response(
        JSON.stringify({ 
          success: true, 
          message: 'Account successfully deleted' 
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );

    } catch (deletionError) {
      console.error('Account deletion failed:', deletionError);
      return new Response(
        JSON.stringify({ 
          error: 'Account deletion failed', 
          details: deletionError.message 
        }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

  } catch (error) {
    console.error('Unexpected error during account deletion:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error', 
        details: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
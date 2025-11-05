import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

console.log("embed-user function booting up");

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    // Supabaseクライアントを初期化
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
      { auth: { persistSession: false } }
    );

    // 認証ヘッダーからユーザー情報を取得
    const authHeader = req.headers.get('Authorization')!;
    const { data: { user } } = await supabase.auth.getUser(authHeader.replace('Bearer ', ''));

    if (!user) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    console.log(`embed-user called for user: ${user.id}`);

    // Check the last updated time to throttle the update
    const { data: featureData, error: featureError } = await supabase
      .from('user_features')
      .select('updated_at')
      .eq('user_id', user.id)
      .single();

    if (featureError && featureError.code !== 'PGRST116') { // PGRST116 means no rows found, which is fine
      throw new Error(`Failed to fetch user_features: ${featureError.message}`);
    }

    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();

    // If features exist and were updated within the last hour, skip the update
    if (featureData && featureData.updated_at > oneHourAgo) {
      return new Response(JSON.stringify({
        success: true,
        user_id: user.id,
        message: "Skipped feature update; last update was too recent.",
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      });
    }

    const { error } = await supabase.rpc('update_user_features', { p_user_id: user.id });

    if (error) {
      console.error(`RPC error for user ${user.id}:`, error.message);
      throw new Error(`Failed to update user features: ${error.message}`);
    }

    const responsePayload = {
      success: true,
      user_id: user.id,
      message: "User features successfully updated.",
    };

    return new Response(JSON.stringify(responsePayload), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
    console.error('Unexpected error in embed-user:', error?.message ?? error);
    return new Response(JSON.stringify({ error: error?.message ?? 'unknown error' }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    });
  }
});

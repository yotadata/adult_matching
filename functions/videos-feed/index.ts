import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey',
}

serve(async (req) => {
  // This is needed if you're planning to invoke your function from a browser.
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // TODO: Implement authentication and logic to switch between random and recommended videos.

    // For now, return 20 dummy video objects.
    const dummyVideos = Array.from({ length: 20 }, (_, i) => ({
      id: `video_${i + 1}`,
      title: `Dummy Video ${i + 1}`,
      description: 'This is a dummy video description.',
      thumbnail_url: `https://placehold.jp/150x150.png?text=Video+${i + 1}`,
      preview_video_url: '',
      source: 'dummy',
      published_at: new Date().toISOString(),
      score: Math.random(),
      reasons: ['dummy reason'],
    }))

    return new Response(JSON.stringify(dummyVideos), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})

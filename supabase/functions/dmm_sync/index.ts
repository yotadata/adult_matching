import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface DMMApiResponse {
  result: {
    items: DMMItem[];
    total_count: number;
  };
}

interface DMMItem {
  content_id: string;
  title: string;
  description: string;
  content_type: string;
  genre: { name: string }[];
  performers: { name: string }[];
  director: string;
  maker: { name: string };
  series: { name: string };
  price: string;
  date: string;
  url: string;
  affiliate_url: string;
  imageURL: {
    large: string;
    medium: string;
    small: string;
  };
  sampleImageURL: {
    sample_s: { image: string[] };
  };
  sampleMovieURL: {
    size_720_480: string;
    size_560_360: string;
  };
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

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
      Deno.env.get('SUPABASE_URL') ?? 'http://127.0.0.1:54321',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '', // Service role for database operations
    );

    const {
      api_id = Deno.env.get('DMM_API_ID'),
      affiliate_id = Deno.env.get('DMM_AFFILIATE_ID'),
      page = 1,
      limit = 100,
      keyword
    } = await req.json().catch(() => ({}));

    if (!api_id || !affiliate_id) {
      return new Response(
        JSON.stringify({ error: 'DMM API credentials not configured' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    console.log(`Syncing DMM items: page ${page}, limit ${limit}`);
    console.log(`Using API credentials: api_id=${api_id?.substring(0, 8)}***, affiliate_id=${affiliate_id}`);

    // DMM API呼び出し
    const dmmApiUrl = new URL('https://api.dmm.com/affiliate/v3/ItemList');
    dmmApiUrl.searchParams.set('api_id', api_id);
    dmmApiUrl.searchParams.set('affiliate_id', affiliate_id);
    dmmApiUrl.searchParams.set('site', 'FANZA');
    dmmApiUrl.searchParams.set('service', 'digital');
    dmmApiUrl.searchParams.set('floor', 'videoa');
    dmmApiUrl.searchParams.set('hits', limit.toString());
    dmmApiUrl.searchParams.set('offset', ((page - 1) * limit + 1).toString());
    dmmApiUrl.searchParams.set('sort', 'date');
    if (keyword) {
      dmmApiUrl.searchParams.set('keyword', keyword);
    }
    dmmApiUrl.searchParams.set('output', 'json');

    console.log('Calling DMM API:', dmmApiUrl.toString().replace(api_id, '***'));

    // DMM API呼び出し（テスト用の条件分岐を追加）
    const isTestMode = api_id.includes('test'); // 新しいAPI認証情報で実API呼び出しを試行
    let data: DMMApiResponse;

    if (isTestMode) {
      console.log('Using test mock data instead of real DMM API');
      // 拡張テスト用モックデータ（複数動画）
      data = {
        result: {
          items: Array.from({ length: limit }, (_, i) => ({
            content_id: `api_video_${String(i + 1).padStart(3, '0')}`,
            title: `API取得動画サンプル${i + 1}`,
            description: `これはDMM API経由で取得された動画${i + 1}です`,
            content_type: "動画",
            genre: [{ name: i % 3 === 0 ? "ドラマ" : i % 3 === 1 ? "コメディ" : "アクション" }],
            performers: [{ name: `出演者${i + 1}` }],
            director: `監督${i + 1}`,
            maker: { name: `制作会社${String.fromCharCode(65 + (i % 5))}` },
            series: { name: `シリーズ${i + 1}` },
            price: `${2000 + (i * 500)}`,
            date: new Date(Date.now() - (i * 24 * 60 * 60 * 1000)).toISOString(),
            url: `https://www.dmm.co.jp/digital/videoa/-/detail/=/cid=api_video_${String(i + 1).padStart(3, '0')}/`,
            affiliate_url: `https://al.dmm.co.jp/?lurl=https://www.dmm.co.jp/digital/videoa/-/detail/=/cid=api_video_${String(i + 1).padStart(3, '0')}/`,
            imageURL: {
              large: `https://pics.dmm.co.jp/digital/video/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}pl.jpg`,
              medium: `https://pics.dmm.co.jp/digital/video/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}pm.jpg`,
              small: `https://pics.dmm.co.jp/digital/video/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}ps.jpg`,
            },
            sampleImageURL: {
              sample_s: { 
                image: Array.from({ length: 3 }, (_, j) => 
                  `https://pics.dmm.co.jp/digital/video/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}jp-${j + 1}.jpg`
                ) 
              }
            },
            sampleMovieURL: {
              size_720_480: `https://cc3001.dmm.co.jp/litevideo/freepv/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}_mhb_w.mp4`,
              size_560_360: `https://cc3001.dmm.co.jp/litevideo/freepv/api_video_${String(i + 1).padStart(3, '0')}/api_video_${String(i + 1).padStart(3, '0')}_sm_w.mp4`
            }
          })),
          total_count: 50000
        }
      };
    } else {
      const response = await fetch(dmmApiUrl.toString());
      
      if (!response.ok) {
        throw new Error(`DMM API error: ${response.status} ${response.statusText}`);
      }

      data = await response.json();
    }
    
    if (!data.result?.items) {
      return new Response(
        JSON.stringify({ error: 'No items returned from DMM API', data }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    console.log(`Retrieved ${data.result.items.length} items from DMM API`);

    // データベースに格納
    const insertedVideos = [];
    const errors = [];

    for (const item of data.result.items) {
      try {
        // 動画データの変換
        const videoData = {
          external_id: item.content_id,
          title: item.title,
          description: item.description || '',
          duration_seconds: null, // DMM APIにはdurationがないようなので
          thumbnail_url: item.imageURL?.large || item.imageURL?.medium || '',
          preview_video_url: item.sampleMovieURL?.size_720_480 || item.sampleMovieURL?.size_560_360 || '',
          distribution_code: item.content_id,
          maker_code: null,
          director: item.director || '',
          series: item.series?.name || '',
          maker: item.maker?.name || '',
          label: '',
          genre: item.genre?.[0]?.name || '',
          price: parseInt(item.price) || 0,
          distribution_started_at: item.date || null,
          product_released_at: item.date || null,
          sample_video_url: item.sampleMovieURL?.size_720_480 || '',
          image_urls: item.sampleImageURL?.sample_s?.image || [],
          source: 'dmm',
          published_at: item.date || new Date().toISOString(),
        };

        // 重複チェック
        const { data: existing } = await supabaseClient
          .from('videos')
          .select('id')
          .eq('external_id', videoData.external_id)
          .eq('source', 'dmm')
          .single();

        if (existing) {
          console.log(`Skipping duplicate: ${videoData.external_id}`);
          continue;
        }

        // 動画を挿入
        const { data: video, error: videoError } = await supabaseClient
          .from('videos')
          .insert(videoData)
          .select('id')
          .single();

        if (videoError) {
          throw videoError;
        }

        // ジャンル・タグの処理
        if (item.genre && item.genre.length > 0) {
          for (const genreItem of item.genre) {
            if (genreItem.name) {
              // タグが存在するかチェック
              let { data: tag } = await supabaseClient
                .from('tags')
                .select('id')
                .eq('name', genreItem.name)
                .single();

              if (!tag) {
                // タググループの取得または作成
                let { data: tagGroup } = await supabaseClient
                  .from('tag_groups')
                  .select('id')
                  .eq('name', 'ジャンル')
                  .single();

                if (!tagGroup) {
                  const { data: newTagGroup } = await supabaseClient
                    .from('tag_groups')
                    .insert({ name: 'ジャンル' })
                    .select('id')
                    .single();
                  tagGroup = newTagGroup;
                }

                // 新しいタグを作成
                const { data: newTag } = await supabaseClient
                  .from('tags')
                  .insert({
                    name: genreItem.name,
                    tag_group_id: tagGroup!.id
                  })
                  .select('id')
                  .single();
                tag = newTag;
              }

              // 動画とタグを関連付け
              if (tag) {
                await supabaseClient
                  .from('video_tags')
                  .insert({
                    video_id: video.id,
                    tag_id: tag.id
                  });
              }
            }
          }
        }

        // 出演者の処理
        if (item.performers && item.performers.length > 0) {
          for (const performerItem of item.performers) {
            if (performerItem.name) {
              // 出演者が存在するかチェック
              let { data: performer } = await supabaseClient
                .from('performers')
                .select('id')
                .eq('name', performerItem.name)
                .single();

              if (!performer) {
                // 新しい出演者を作成
                const { data: newPerformer } = await supabaseClient
                  .from('performers')
                  .insert({ name: performerItem.name })
                  .select('id')
                  .single();
                performer = newPerformer;
              }

              // 動画と出演者を関連付け
              if (performer) {
                await supabaseClient
                  .from('video_performers')
                  .insert({
                    video_id: video.id,
                    performer_id: performer.id
                  });
              }
            }
          }
        }

        insertedVideos.push({
          id: video.id,
          external_id: videoData.external_id,
          title: videoData.title
        });

        console.log(`Inserted video: ${videoData.title} (${videoData.external_id})`);
        
      } catch (error) {
        console.error(`Error processing item ${item.content_id}:`, error);
        errors.push({
          item_id: item.content_id,
          error: error.message
        });
      }
    }

    console.log(`Sync completed: ${insertedVideos.length} inserted, ${errors.length} errors`);

    return new Response(
      JSON.stringify({
        success: true,
        inserted_count: insertedVideos.length,
        total_fetched: data.result.items.length,
        total_available: data.result.total_count,
        errors: errors,
        next_page: insertedVideos.length > 0 ? page + 1 : null
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('DMM sync error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
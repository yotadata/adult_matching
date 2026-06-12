import { ImageResponse } from 'next/og';
import { supabase } from '@/lib/supabase';

export const runtime = 'edge';
export const alt = '性癖ラボ リスト';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

type Video = {
  thumbnail_url: string | null;
};

type ListData = {
  title: string | null;
  display_name: string | null;
  view_count: number;
  like_count: number;
  videos: Video[];
};

export default async function Image({ params }: { params: { token: string } }) {
  const { data } = await supabase.rpc('get_public_list_data', { p_token: params.token });

  if (!data || data.error === 'not_found') {
    // フォールバック: シンプルなデフォルト画像
    return new ImageResponse(
      (
        <div style={{
          width: '1200px', height: '630px',
          background: 'linear-gradient(135deg, #0d1117 0%, #161b22 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ color: '#656d76', fontSize: '32px', fontFamily: 'sans-serif' }}>
            性癖ラボ
          </span>
        </div>
      ),
      { width: 1200, height: 630 }
    );
  }

  const list = data as ListData;
  const title = list.title ?? (list.display_name ? `${list.display_name}のお気に入りリスト` : 'お気に入りリスト');
  const thumbs = (list.videos ?? [])
    .map((v) => v.thumbnail_url)
    .filter(Boolean)
    .slice(0, 5) as string[];

  const videoCount = (list.videos ?? []).length;
  const viewCount = list.view_count ?? 0;
  const likeCount = list.like_count ?? 0;

  // サムネイルを1〜5枚の横並びコラージュとして表示
  const thumbCount = thumbs.length;
  const thumbWidth = thumbCount > 0 ? Math.floor(840 / thumbCount) : 840;

  return new ImageResponse(
    (
      <div style={{
        width: '1200px',
        height: '630px',
        background: '#0d1117',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'sans-serif',
        overflow: 'hidden',
      }}>
        {/* サムネイルエリア（上部 約70%） */}
        <div style={{
          display: 'flex',
          width: '1200px',
          height: '440px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          {thumbCount > 0 ? (
            thumbs.map((url, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={i}
                src={url}
                width={thumbWidth}
                height={440}
                style={{
                  objectFit: 'cover',
                  flexShrink: 0,
                  borderRight: i < thumbCount - 1 ? '2px solid #0d1117' : 'none',
                }}
                alt=""
              />
            ))
          ) : (
            <div style={{
              width: '1200px',
              height: '440px',
              background: 'linear-gradient(135deg, #161b22 0%, #21262d 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ color: '#484f58', fontSize: '24px' }}>No Image</span>
            </div>
          )}
          {/* グラデーションオーバーレイ（下部フェード） */}
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: '160px',
            background: 'linear-gradient(to bottom, transparent, #0d1117)',
            display: 'flex',
          }} />
        </div>

        {/* 情報エリア（下部） */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          padding: '0 48px 32px',
          gap: '12px',
          marginTop: '-40px',
          position: 'relative',
          zIndex: 1,
        }}>
          {/* タイトル */}
          <div style={{
            color: '#e6edf3',
            fontSize: '40px',
            fontWeight: 900,
            lineHeight: 1.2,
            display: 'flex',
            maxWidth: '1100px',
          }}>
            {title.length > 30 ? `${title.slice(0, 30)}…` : title}
          </div>

          {/* 統計 */}
          <div style={{
            display: 'flex',
            gap: '24px',
            alignItems: 'center',
          }}>
            <span style={{ color: '#8b949e', fontSize: '22px' }}>
              {videoCount.toLocaleString()}作品
            </span>
            {viewCount > 0 && (
              <span style={{ color: '#8b949e', fontSize: '22px' }}>
                👁 {viewCount.toLocaleString()}
              </span>
            )}
            {likeCount > 0 && (
              <span style={{ color: '#8b949e', fontSize: '22px' }}>
                ❤️ {likeCount.toLocaleString()}
              </span>
            )}
            <span style={{
              marginLeft: 'auto',
              color: '#484f58',
              fontSize: '18px',
            }}>
              seihekilab.com
            </span>
          </div>
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}

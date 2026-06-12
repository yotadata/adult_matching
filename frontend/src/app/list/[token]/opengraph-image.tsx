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
    return new ImageResponse(
      (
        <div style={{
          width: '1200px', height: '630px',
          background: '#0d1117',
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
  const displayName = list.display_name ?? '';
  const title = list.title ?? (displayName ? `${displayName}のお気に入りリスト` : 'お気に入りリスト');
  const thumbs = (list.videos ?? [])
    .map((v) => v.thumbnail_url)
    .filter(Boolean)
    .slice(0, 5) as string[];

  const videoCount = (list.videos ?? []).length;
  const viewCount = list.view_count ?? 0;
  const likeCount = list.like_count ?? 0;

  // サムネイルを均等割りで横並び（足りない分は最後の画像で埋める）
  const cols = Math.max(thumbs.length, 1);
  const colWidth = Math.floor(1200 / cols);
  // 均等割りで残り px を最後のカラムに割り当て
  const lastColWidth = 1200 - colWidth * (cols - 1);

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
        {/* サムネイルエリア */}
        <div style={{
          display: 'flex',
          width: '1200px',
          height: '450px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          {thumbs.length > 0 ? (
            thumbs.map((url, i) => {
              const w = i === thumbs.length - 1 ? lastColWidth : colWidth;
              return (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={i}
                  src={url}
                  width={w}
                  height={450}
                  style={{
                    objectFit: 'cover',
                    flexShrink: 0,
                    borderRight: i < thumbs.length - 1 ? '2px solid #0d1117' : 'none',
                  }}
                  alt=""
                />
              );
            })
          ) : (
            <div style={{
              width: '1200px', height: '450px',
              background: '#161b22',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ color: '#484f58', fontSize: '24px' }}>No Image</span>
            </div>
          )}
          {/* 下部グラデーションオーバーレイ */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: '200px',
            background: 'linear-gradient(to bottom, transparent, #0d1117)',
            display: 'flex',
          }} />
        </div>

        {/* 情報エリア */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          padding: '0 48px 28px',
          gap: '10px',
          marginTop: '-60px',
          position: 'relative',
          zIndex: 1,
        }}>
          {/* タイトル */}
          <div style={{
            color: '#e6edf3',
            fontSize: '42px',
            fontWeight: 900,
            lineHeight: 1.2,
            display: 'flex',
          }}>
            {title.length > 28 ? `${title.slice(0, 28)}…` : title}
          </div>

          {/* 統計 + ユーザー名 */}
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
            <span style={{ color: '#8b949e', fontSize: '20px' }}>
              {videoCount.toLocaleString()}作品
            </span>
            {viewCount > 0 && (
              <span style={{ color: '#8b949e', fontSize: '20px' }}>
                👁 {viewCount.toLocaleString()}
              </span>
            )}
            {likeCount > 0 && (
              <span style={{ color: '#8b949e', fontSize: '20px' }}>
                ❤️ {likeCount.toLocaleString()}
              </span>
            )}
            {displayName && (
              <span style={{ marginLeft: 'auto', color: '#656d76', fontSize: '18px' }}>
                by {displayName}
              </span>
            )}
          </div>
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}

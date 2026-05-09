import { ImageResponse } from 'next/og'
import { NextRequest } from 'next/server'
import { QUIZ_TYPES, AXIS_META, QuizTypeKey, Axis } from '@/app/quiz/data'

export const runtime = 'edge'

const AXES: Axis[] = ['ds', 'pe', 'nx', 'cw']

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url)
  const typeKey = (searchParams.get('type') ?? 'senc') as QuizTypeKey
  const scoresRaw = searchParams.get('scores') ?? '{}'

  let scores: Record<string, number> = {}
  try { scores = JSON.parse(decodeURIComponent(scoresRaw)) } catch {}

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['senc']
  const charImageUrl = `${origin}/quiz/${typeKey}.png`

  return new ImageResponse(
    (
      <div
        style={{
          width: '1200px',
          height: '630px',
          background: 'linear-gradient(135deg, #1a0d2e 0%, #2a1020 60%, #1e0d1a 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 80px',
          gap: '64px',
          fontFamily: 'sans-serif',
        }}
      >
        {/* 左: キャラクター */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={charImageUrl}
            width={280}
            height={280}
            style={{ objectFit: 'contain', filter: 'drop-shadow(0 8px 24px rgba(0,0,0,0.4))' }}
            alt=""
          />
        </div>

        {/* 右: テキスト + スコアバー */}
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
          {/* ラベル */}
          <p style={{
            color: 'rgba(216,180,254,0.6)',
            fontSize: '16px',
            letterSpacing: '0.25em',
            textTransform: 'uppercase',
            margin: '0 0 12px 0',
          }}>
            偏愛16診断
          </p>

          {/* タイプキーバッジ */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '14px' }}>
            {typeKey.toUpperCase().split('').map((c, i) => (
              <div
                key={i}
                style={{
                  background: quizType.color,
                  color: quizType.accent,
                  fontSize: '15px',
                  fontWeight: 900,
                  padding: '3px 12px',
                  borderRadius: '100px',
                }}
              >
                {c}
              </div>
            ))}
          </div>

          {/* タイプ名 */}
          <h1 style={{
            color: 'white',
            fontSize: '60px',
            fontWeight: 900,
            margin: '0 0 8px 0',
            lineHeight: 1.1,
          }}>
            {quizType.name}
          </h1>

          {/* タグライン */}
          <p style={{
            color: quizType.color,
            fontSize: '20px',
            fontWeight: 700,
            margin: '0 0 28px 0',
          }}>
            {quizType.tagline}
          </p>

          {/* スコアバー */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {AXES.map((axis) => {
              const pct = typeof scores[axis] === 'number' ? scores[axis] : 50
              const meta = AXIS_META[axis]
              const color = pct >= 50 ? meta.colorHigh : meta.colorLow
              return (
                <div key={axis} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '13px' }}>{meta.labelHigh}</span>
                    <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '13px' }}>{meta.labelLow}</span>
                  </div>
                  <div style={{
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '100px',
                    height: '8px',
                    overflow: 'hidden',
                    position: 'relative',
                    display: 'flex',
                  }}>
                    <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', background: 'rgba(255,255,255,0.2)' }} />
                    {pct >= 50
                      ? <div style={{ position: 'absolute', right: '50%', top: 0, bottom: 0, width: `${(pct - 50) * 2}%`, background: color, borderRadius: '100px' }} />
                      : <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: `${(50 - pct) * 2}%`, background: color, borderRadius: '100px' }} />
                    }
                  </div>
                </div>
              )
            })}
          </div>

          {/* フッター */}
          <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: '14px', margin: '20px 0 0 0' }}>
            seihekilab.com
          </p>
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  )
}

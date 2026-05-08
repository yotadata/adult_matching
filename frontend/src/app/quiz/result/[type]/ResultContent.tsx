'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { toPng } from 'html-to-image';
import { QUIZ_TYPES, AXIS_META, QuizTypeKey, QuizType, Axis } from '../../data';
import { trackEvent } from '@/lib/analytics';

const CTA_SHOWN_KEY = 'quiz_male_cta_shown';
const AXES: Axis[] = ['ds', 'nx', 'pe', 'hl'];

// ─── シェアカード（html-to-image でキャプチャ用） ────────────────────────────
function ShareCard({
  typeKey,
  quizType,
  axes,
  cardRef,
}: {
  typeKey: QuizTypeKey;
  quizType: QuizType;
  axes: { axis: Axis; pct: number }[];
  cardRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div
      ref={cardRef}
      style={{
        position: 'fixed',
        top: '-9999px',
        left: 0,
        width: '750px',
        background: 'linear-gradient(135deg, #1a0d2e 0%, #2a1020 60%, #1e0d1a 100%)',
        padding: '48px',
        fontFamily: 'sans-serif',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <p style={{ color: 'rgba(216,180,254,0.6)', fontSize: '13px', letterSpacing: '0.3em', textTransform: 'uppercase', marginBottom: '20px' }}>
        性癖16タイプ診断
      </p>

      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`/quiz/${typeKey}.png`}
        alt={quizType.name}
        style={{ width: '240px', height: '240px', objectFit: 'contain', filter: 'drop-shadow(0 6px 20px rgba(0,0,0,0.5))', marginBottom: '20px' }}
        crossOrigin="anonymous"
      />

      <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
        {typeKey.toUpperCase().split('').map((c, i) => (
          <span key={i} style={{ background: quizType.color, color: quizType.accent, fontSize: '13px', fontWeight: 900, padding: '3px 12px', borderRadius: '100px' }}>
            {c}
          </span>
        ))}
      </div>

      <h2 style={{ color: 'white', fontSize: '48px', fontWeight: 900, margin: '0 0 6px', textAlign: 'center', lineHeight: 1.1 }}>
        {quizType.name}
      </h2>
      <p style={{ color: quizType.color, fontSize: '18px', fontWeight: 700, margin: '0 0 28px', textAlign: 'center' }}>
        {quizType.tagline}
      </p>

      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '28px' }}>
        {axes.map(({ axis, pct }) => {
          const meta = AXIS_META[axis];
          const color = pct >= 50 ? meta.colorHigh : meta.colorLow;
          return (
            <div key={axis} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: '13px', fontWeight: 700 }}>{meta.labelHigh}</span>
                <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: '13px', fontWeight: 700 }}>{meta.labelLow}</span>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '100px', height: '10px', overflow: 'hidden' }}>
                <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '100px' }} />
              </div>
              <p style={{ color, fontSize: '12px', fontWeight: 700, textAlign: 'right', margin: 0 }}>{pct}%</p>
            </div>
          );
        })}
      </div>

      <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: '13px', margin: 0 }}>seihekilab.com</p>
    </div>
  );
}

// ─── 「画像で保存/シェア」ボタン ────────────────────────────────────────────
function SaveImageButton({
  cardRef,
  typeKey,
  quizType,
}: {
  cardRef: React.RefObject<HTMLDivElement | null>;
  typeKey: QuizTypeKey;
  quizType: QuizType;
}) {
  const [loading, setLoading] = useState(false);

  const handleSave = async () => {
    if (!cardRef.current || loading) return;
    setLoading(true);
    trackEvent('quiz_save_image', { type: typeKey });
    try {
      const dataUrl = await toPng(cardRef.current, { pixelRatio: 2, cacheBust: true });
      const blob = await (await fetch(dataUrl)).blob();
      const file = new File([blob], `seiheki_${typeKey}.png`, { type: 'image/png' });

      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({
          files: [file],
          title: `私の性癖16タイプは「${quizType.name}」でした！`,
          text: quizType.tagline,
        });
        trackEvent('quiz_image_shared', { type: typeKey, method: 'webshare' });
      } else {
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `seiheki_${typeKey}.png`;
        a.click();
        trackEvent('quiz_image_shared', { type: typeKey, method: 'download' });
      }
    } catch (e) {
      console.error('画像生成失敗:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleSave}
      disabled={loading}
      className="w-full rounded-2xl py-4 font-black flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-all"
      style={
        loading
          ? { background: '#e8c9a0', color: '#c8a080', cursor: 'not-allowed' }
          : { background: 'linear-gradient(90deg, #6c3483, #e84393)', color: '#fff', boxShadow: '0 4px 0 #4a235a' }
      }
    >
      {loading ? '生成中…' : '🖼️ 画像を保存してシェア'}
    </button>
  );
}

// ─── ポップアップ ────────────────────────────────────────────────────────────
function MaleCTAModal({ typeKey, onClose }: { typeKey: QuizTypeKey; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center pb-6 px-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-3xl p-6"
        style={{ background: 'linear-gradient(135deg, #1a0d2e, #2a1020)', boxShadow: '0 4px 0 #0d0616, 0 8px 40px rgba(0,0,0,0.5)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-start mb-4">
          <p className="text-xs font-bold tracking-widest text-purple-300/70 uppercase">あなたへのおすすめ</p>
          <button onClick={onClose} className="text-white/40 hover:text-white/80 transition-colors text-lg leading-none -mt-0.5" aria-label="閉じる">✕</button>
        </div>
        <p className="text-xl font-black text-white mb-2">あなたの性癖に合う動画、見つかるかも</p>
        <p className="text-sm text-white/60 mb-5">AIがあなたの好みを学習して、刺さる動画だけをおすすめ。スワイプして探してみよう。</p>
        <Link
          href="/"
          className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: 'linear-gradient(90deg, #9333ea, #ec4899)', boxShadow: '0 4px 0 #6b21a8' }}
          onClick={() => trackEvent('quiz_swipe_cta_click', { type: typeKey })}
        >
          性癖ラボを試してみる 🔥
        </Link>
        <button onClick={onClose} className="w-full mt-3 text-sm font-bold text-white/30 hover:text-white/60 transition-colors">
          今はいい
        </button>
      </div>
    </div>
  );
}

// ─── メインコンテンツ ─────────────────────────────────────────────────────────
export function ResultContent({ typeKey }: { typeKey: QuizTypeKey }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const shareCardRef = useRef<HTMLDivElement>(null);
  const [showCTA, setShowCTA] = useState(false);

  const gender = searchParams.get('gender') ?? 'other';
  const isMale = gender === 'male';

  const rawScores = (() => {
    try { return JSON.parse(decodeURIComponent(searchParams.get('scores') ?? '{}')); }
    catch { return {}; }
  })();

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['sneh'];
  const shareText = `私の性癖16タイプは「${quizType.name}」でした！${quizType.tagline}\n\nあなたは？👇`;
  const scoresParam = searchParams.get('scores') ?? '';
  const shareUrl = `https://seihekilab.com/quiz/result/${typeKey}?scores=${scoresParam}`;

  const axes: { axis: Axis; pct: number }[] = AXES.map((axis) => ({
    axis,
    pct: typeof rawScores[axis] === 'number' ? rawScores[axis] : 50,
  }));

  useEffect(() => {
    trackEvent('quiz_result_view', { type: typeKey, gender });
  }, [typeKey, gender]);

  useEffect(() => {
    if (!isMale) return;
    try { if (localStorage.getItem(CTA_SHOWN_KEY)) return; } catch {}
    const timer = setTimeout(() => {
      setShowCTA(true);
      trackEvent('quiz_male_cta_shown', { type: typeKey });
      try { localStorage.setItem(CTA_SHOWN_KEY, '1'); } catch {}
    }, 2000);
    return () => clearTimeout(timer);
  }, [isMale, typeKey]);

  const closeCTA = () => {
    setShowCTA(false);
    trackEvent('quiz_male_cta_dismissed', { type: typeKey });
  };

  const shareToX = () => {
    trackEvent('quiz_share', { method: 'x', type: typeKey });
    window.open(
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}&hashtags=性癖16タイプ診断`,
      '_blank', 'noopener'
    );
  };
  const shareToLine = () => {
    trackEvent('quiz_share', { method: 'line', type: typeKey });
    window.open(`https://line.me/R/msg/text/?${encodeURIComponent(`${shareText}\n${shareUrl}`)}`, '_blank', 'noopener');
  };
  const copyLink = async () => {
    trackEvent('quiz_share', { method: 'copy_link', type: typeKey });
    await navigator.clipboard.writeText(shareUrl);
  };

  return (
    <div className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8">
      {showCTA && <MaleCTAModal typeKey={typeKey} onClose={closeCTA} />}

      {/* キャプチャ用シェアカード（画面外） */}
      <ShareCard typeKey={typeKey} quizType={quizType} axes={axes} cardRef={shareCardRef} />

      <p className="text-[11px] font-black tracking-[0.3em] text-[#b5541a]/50 uppercase mb-5">✦ 診断結果 ✦</p>

      {/* メインカード */}
      <div
        className="w-full max-w-sm rounded-3xl overflow-hidden mb-6"
        style={{
          background: '#fffdf5',
          border: '2px solid #e0c090',
          outline: '2px dashed rgba(180,120,60,0.28)',
          outlineOffset: '-8px',
          boxShadow: '0 4px 0 #c8946a, 0 8px 28px rgba(100,50,0,0.12)',
        }}
      >
        {/* キャラクターヘッダー */}
        <div
          className="relative h-72 flex items-end justify-center pb-4 overflow-hidden"
          style={{ background: `${quizType.color}28` }}
        >
          <div
            className="absolute inset-3 rounded-2xl"
            style={{ border: `2px dashed ${quizType.color}70`, background: `${quizType.color}10` }}
          />
          <div
            className="absolute bottom-0 left-0 right-0 h-10"
            style={{ background: '#fffdf5', clipPath: 'ellipse(65% 100% at 50% 100%)' }}
          />
          <Image
            src={`/quiz/${typeKey}.png`}
            alt={quizType.name}
            width={260}
            height={260}
            className="relative object-contain"
            style={{ filter: 'drop-shadow(0 6px 18px rgba(0,0,0,0.18))' }}
          />
        </div>

        <div className="px-6 pt-4 pb-6">
          <div className="flex items-center gap-1.5 mb-3">
            {typeKey.toUpperCase().split('').map((c, i) => (
              <span
                key={i}
                className="text-[10px] font-black px-2 py-0.5 rounded-full"
                style={{ background: `${quizType.color}28`, color: quizType.accent, border: `1px solid ${quizType.color}60` }}
              >
                {c}
              </span>
            ))}
          </div>

          <h2 className="text-[28px] font-black text-[#3d1a00] leading-tight mb-1">{quizType.name}</h2>
          <p className="text-sm font-bold mb-4" style={{ color: quizType.accent }}>{quizType.tagline}</p>
          <p className="text-sm leading-relaxed text-[#6b4423] mb-6">{quizType.description}</p>

          <div className="space-y-4 pt-5" style={{ borderTop: '2px dashed rgba(180,120,60,0.25)' }}>
            <p className="text-[10px] font-black tracking-widest text-[#b5541a]/50 uppercase">✦ あなたの傾向 ✦</p>
            {axes.map(({ axis, pct }) => {
              const meta = AXIS_META[axis];
              const isHigh = pct >= 50;
              const color = isHigh ? meta.colorHigh : meta.colorLow;
              let degreeLabel: string;
              if (pct >= 80)      degreeLabel = meta.degreesHigh[0];
              else if (pct >= 60) degreeLabel = meta.degreesHigh[1];
              else if (pct >= 40) degreeLabel = 'ニュートラル';
              else if (pct >= 20) degreeLabel = meta.degreesLow[1];
              else               degreeLabel = meta.degreesLow[0];

              return (
                <div key={axis}>
                  <div className="flex justify-between items-center mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-black text-[#8b5e3c]">{meta.labelHigh}</span>
                      <span className="text-[10px] text-[#c8a880]">⇄</span>
                      <span className="text-[11px] font-black text-[#8b5e3c]">{meta.labelLow}</span>
                    </div>
                    <span className="text-[11px] font-black" style={{ color }}>{degreeLabel}</span>
                  </div>
                  <div className="relative h-3 rounded-full overflow-hidden" style={{ background: 'rgba(180,120,60,0.12)' }}>
                    <div className="absolute left-1/2 top-0 bottom-0 w-px z-10" style={{ background: 'rgba(180,120,60,0.3)' }} />
                    <div className="absolute left-0 top-0 bottom-0 rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
                  </div>
                  <div className="flex justify-between text-[10px] text-[#b5541a]/40 mt-0.5 font-bold">
                    <span>{meta.labelHigh}</span>
                    <span>{pct}%</span>
                    <span>{meta.labelLow}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] text-[#d4a574]/50 font-bold tracking-widest mt-5 text-right">✦ 性癖16タイプ診断 ✦</p>
        </div>
      </div>

      {/* シェアボタン */}
      <div className="w-full max-w-sm space-y-3 mb-6">
        <SaveImageButton cardRef={shareCardRef} typeKey={typeKey} quizType={quizType} />
        <button
          onClick={shareToX}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#1a1a1a', boxShadow: '0 4px 0 #000', border: '2px solid #333' }}
        >
          <span className="text-lg">𝕏</span> ポストして友だちに教える
        </button>
        <button
          onClick={shareToLine}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#06C755', boxShadow: '0 4px 0 #04a344', border: '2px solid #08e060' }}
        >
          <span className="text-lg">💬</span> LINEで送る
        </button>
        <button
          onClick={copyLink}
          className="w-full rounded-2xl py-4 font-black flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#fffdf5', border: '2px dashed rgba(180,120,60,0.4)', boxShadow: '0 3px 0 #c8946a', color: '#3d1a00' }}
        >
          🔗 リンクをコピー
        </button>
      </div>

      {/* 男性向けCTA（インライン） */}
      {isMale && (
        <div
          className="w-full max-w-sm rounded-3xl p-6 mb-5"
          style={{ background: 'linear-gradient(135deg, #1a0d2e, #2a1020)', border: '2px solid #3d1a5e', boxShadow: '0 4px 0 #0d0616, 0 8px 24px rgba(0,0,0,0.3)' }}
        >
          <p className="text-xs font-bold tracking-widest text-purple-300/60 uppercase mb-2">✦ あなたへのおすすめ ✦</p>
          <p className="text-xl font-black text-white mb-2">あなたの性癖に合う動画、見つかるかも</p>
          <p className="text-sm text-white/55 mb-4">AIがあなたの好みを学習して、刺さる動画だけをおすすめ。スワイプして探してみよう。</p>
          <Link
            href="/"
            className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] active:translate-y-[1px] transition-transform"
            style={{ background: 'linear-gradient(90deg, #7b2d8b, #c4337a)', boxShadow: '0 4px 0 #5a1a6b', border: '2px solid #9b3dab' }}
            onClick={() => trackEvent('quiz_swipe_cta_click', { type: typeKey })}
          >
            性癖ラボを試してみる ✦
          </Link>
        </div>
      )}

      <button onClick={() => router.push('/quiz')} className="text-sm font-bold text-[#b5541a]/50 underline underline-offset-4">
        もう一度診断する
      </button>
    </div>
  );
}

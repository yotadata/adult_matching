'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { toPng } from 'html-to-image';
import QRCode from 'qrcode';
import { QUIZ_TYPES, AXIS_META, COMPATIBILITY, COMPATIBILITY_LABELS, QuizTypeKey, QuizType, Axis } from '../../data';
import { trackEvent } from '@/lib/analytics';

const CTA_SHOWN_KEY = 'quiz_male_cta_shown';
const AXES: Axis[] = ['ds', 'pe', 'nx', 'cw'];

// ─── シェアカード（html-to-image でキャプチャ用） ────────────────────────────
function ShareCard({
  typeKey,
  quizType,
  axes,
  qrDataUrl,
  charDataUrl,
  cardRef,
}: {
  typeKey: QuizTypeKey;
  quizType: QuizType;
  axes: { axis: Axis; pct: number }[];
  qrDataUrl: string;
  charDataUrl: string;
  cardRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div
      ref={cardRef}
      style={{
        position: 'relative',
        width: '750px',
        height: '750px',
        background: 'linear-gradient(170deg, #0d0b08 0%, #1a1510 100%)',
        fontFamily: 'sans-serif',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* キャラクター画像エリア */}
      <div style={{ height: '450px', background: `${quizType.color}38`, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', flexShrink: 0 }}>
        <div style={{ position: 'absolute', inset: 0, background: `radial-gradient(ellipse at center, ${quizType.color}22 0%, transparent 70%)` }} />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={charDataUrl || `/quiz/${typeKey}.png`}
          alt={quizType.name}
          style={{ width: '380px', height: '380px', objectFit: 'contain', filter: 'drop-shadow(0 12px 32px rgba(0,0,0,0.85))', position: 'relative' }}
        />
        {/* ヘッダーバー */}
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, padding: '12px 22px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ color: 'rgba(180,150,80,0.65)', fontSize: '11px', letterSpacing: '0.22em', textTransform: 'uppercase', margin: 0 }}>性癖16タイプ分析</p>
          <div style={{ display: 'flex', gap: '5px' }}>
            {typeKey.toUpperCase().split('').map((c, i) => (
              <span key={i} style={{ background: `${quizType.color}30`, color: quizType.color, fontSize: '11px', fontWeight: 900, padding: '3px 10px', borderRadius: '100px', border: `1px solid ${quizType.color}55` }}>{c}</span>
            ))}
          </div>
        </div>
        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '80px', background: 'linear-gradient(to bottom, transparent, #0d0b08)' }} />
        {/* QRコード（画像エリア右下） */}
        {qrDataUrl && (
          <div style={{ position: 'absolute', bottom: '14px', right: '16px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={qrDataUrl} alt="QR" style={{ width: '64px', height: '64px', borderRadius: '6px', background: 'white', padding: '4px', display: 'block' }} />
            <p style={{ color: 'rgba(180,150,80,0.55)', fontSize: '9px', margin: 0, letterSpacing: '0.05em' }}>seihekilab.com</p>
          </div>
        )}
      </div>

      {/* テキストエリア */}
      <div style={{ padding: '18px 28px 20px', display: 'flex', flexDirection: 'column', gap: '12px', flex: 1 }}>
        {/* 名前・タグライン */}
        <div>
          <h2 style={{ color: '#f0e6d3', fontSize: '38px', fontWeight: 900, margin: '0 0 5px', lineHeight: 1.1 }}>{quizType.name}</h2>
          <p style={{ color: quizType.color, fontSize: '15px', fontWeight: 700, margin: 0, lineHeight: 1.4 }}>{quizType.tagline}</p>
        </div>

        {/* 説明文 */}
        <p style={{ color: 'rgba(200,180,140,0.65)', fontSize: '12px', lineHeight: 1.7, margin: 0 }}>
          {quizType.description}
        </p>

        {/* 軸バー */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '9px' }}>
          {axes.map(({ axis, pct }) => {
            const meta = AXIS_META[axis];
            const isHigh = pct >= 50;
            const color = isHigh ? meta.colorHigh : meta.colorLow;
            const label = isHigh ? meta.labelHigh : meta.labelLow;
            const barWidth = Math.abs(pct - 50) * 2;
            return (
              <div key={axis} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ color: 'rgba(200,180,140,0.5)', fontSize: '11px', fontWeight: 700, width: '34px', textAlign: 'right', flexShrink: 0 }}>{meta.labelHigh}</span>
                <div style={{ flex: 1, background: 'rgba(180,150,80,0.1)', borderRadius: '100px', height: '7px', overflow: 'hidden', position: 'relative' }}>
                  <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', background: 'rgba(180,150,80,0.25)' }} />
                  {isHigh
                    ? <div style={{ position: 'absolute', right: '50%', top: 0, bottom: 0, width: `${barWidth}%`, background: color, borderRadius: '100px' }} />
                    : <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: `${barWidth}%`, background: color, borderRadius: '100px' }} />
                  }
                </div>
                <span style={{ color: 'rgba(200,180,140,0.5)', fontSize: '11px', fontWeight: 700, width: '34px', flexShrink: 0 }}>{meta.labelLow}</span>
                <span style={{ color, fontSize: '11px', fontWeight: 900, width: '44px', textAlign: 'right', flexShrink: 0 }}>{label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── 「画像で保存/シェア」ボタン ────────────────────────────────────────────
function SaveImageButton({
  cardRef,
  typeKey,
  quizType,
  charDataUrl,
  onCharDataUrlReady,
}: {
  cardRef: React.RefObject<HTMLDivElement | null>;
  typeKey: QuizTypeKey;
  quizType: QuizType;
  charDataUrl: string;
  onCharDataUrlReady: (url: string) => void;
}) {
  const [loading, setLoading] = useState(false);

  const handleSave = async () => {
    if (!cardRef.current || loading) return;
    setLoading(true);
    trackEvent('quiz_save_image', { type: typeKey });
    try {
      // キャラ画像のデータURLを確保（未取得なら今すぐfetch）
      let resolvedCharDataUrl = charDataUrl;
      if (!resolvedCharDataUrl) {
        resolvedCharDataUrl = await fetch(`/quiz/${typeKey}.png`)
          .then((r) => r.blob())
          .then((blob) => new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.readAsDataURL(blob);
          }));
        onCharDataUrlReady(resolvedCharDataUrl);
      }

      // oncloneでクローン後のimg要素にデータURLを直接差し込む
      const capturedDataUrl = await toPng(cardRef.current, {
        pixelRatio: 2,
        cacheBust: false,
        onclone: (_doc, el) => {
          const img = el.querySelector('img') as HTMLImageElement | null;
          if (img) img.src = resolvedCharDataUrl;
        },
      });

      const blob = await (await fetch(capturedDataUrl)).blob();
      const file = new File([blob], `seiheki_${typeKey}.png`, { type: 'image/png' });

      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file] });
        trackEvent('quiz_image_shared', { type: typeKey, method: 'webshare' });
      } else {
        const a = document.createElement('a');
        a.href = capturedDataUrl;
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
          ? { background: '#e0c090', color: '#c8a080', cursor: 'not-allowed' }
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
          target="_blank"
          rel="noopener noreferrer"
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
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [charDataUrl, setCharDataUrl] = useState('');

  useEffect(() => {
    QRCode.toDataURL('https://seihekilab.com/quiz', { width: 128, margin: 1, color: { dark: '#000000', light: '#ffffff' } })
      .then(setQrDataUrl)
      .catch(() => {});
  }, []);

  // キャラ画像をデータURLとして事前取得（html-to-imageがネットワーク未取得の画像を空で描画するのを防ぐ）
  useEffect(() => {
    fetch(`/quiz/${typeKey}.png`)
      .then((r) => r.blob())
      .then((blob) => new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
      }))
      .then(setCharDataUrl)
      .catch(() => {});
  }, [typeKey]);

  const gender = searchParams.get('gender') ?? 'other';
  const isMale = gender === 'male';

  const rawScores = (() => {
    try { return JSON.parse(decodeURIComponent(searchParams.get('scores') ?? '{}')); }
    catch { return {}; }
  })();

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['senc'];
  const shareText = `私の性癖16タイプは「${quizType.name}」でした！`;
  const scoresParam = searchParams.get('scores') ?? '';
  const shareUrl = `https://seihekilab.com/quiz/result/${typeKey}?scores=${encodeURIComponent(scoresParam)}`;

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
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}&hashtags=性癖16タイプ分析,性癖ラボ`,
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

      {/* キャプチャ用シェアカード（画面外非表示） */}
      <div style={{ position: 'absolute', left: '-9999px', top: 0, overflow: 'hidden', pointerEvents: 'none' }}>
        <ShareCard typeKey={typeKey} quizType={quizType} axes={axes} qrDataUrl={qrDataUrl} charDataUrl={charDataUrl} cardRef={shareCardRef} />
      </div>

      <p className="text-[11px] font-black tracking-[0.3em] uppercase mb-5" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ 診断結果 ✦</p>

      {/* メインカード */}
      <div
        className="w-full max-w-sm rounded-3xl overflow-hidden mb-6"
        style={{
          background: 'rgba(28,24,18,0.9)',
          border: '1px solid rgba(180,150,80,0.35)',
          boxShadow: '0 4px 0 rgba(0,0,0,0.5), 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(180,150,80,0.15)',
          backdropFilter: 'blur(8px)',
        }}
      >
        {/* キャラクターヘッダー */}
        <div
          className="relative h-72 flex items-center justify-center overflow-hidden"
          style={{ background: `${quizType.color}55` }}
        >
          <div
            className="absolute inset-3 rounded-2xl"
            style={{ border: `1px solid ${quizType.color}50` }}
          />
          <Image
            src={`/quiz/${typeKey}.png`}
            alt={quizType.name}
            width={260}
            height={260}
            className="relative object-contain"
            style={{ filter: 'drop-shadow(0 6px 24px rgba(0,0,0,0.6))' }}
          />
        </div>

        <div className="px-6 pt-4 pb-6">
          <div className="flex items-center gap-1.5 mb-3">
            {typeKey.toUpperCase().split('').map((c, i) => (
              <span
                key={i}
                className="text-[10px] font-black px-2 py-0.5 rounded-full"
                style={{ background: `${quizType.color}25`, color: quizType.color, border: `1px solid ${quizType.color}60` }}
              >
                {c}
              </span>
            ))}
          </div>

          <h2 className="text-[28px] font-black leading-tight mb-1" style={{ color: '#f0e6d3' }}>{quizType.name}</h2>
          <p className="text-sm font-bold mb-4" style={{ color: quizType.color }}>{quizType.tagline}</p>
          <p className="text-sm leading-relaxed mb-6" style={{ color: 'rgba(200,180,140,0.7)' }}>{quizType.description}</p>

          <div className="space-y-4 pt-5" style={{ borderTop: '1px solid rgba(180,150,80,0.2)' }}>
            <p className="text-[10px] font-black tracking-widest uppercase" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ あなたの傾向 ✦</p>
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
                      <span className="text-[11px] font-black" style={{ color: 'rgba(200,180,140,0.7)' }}>{meta.labelHigh}</span>
                      <span className="text-[10px]" style={{ color: 'rgba(180,150,80,0.5)' }}>⇄</span>
                      <span className="text-[11px] font-black" style={{ color: 'rgba(200,180,140,0.7)' }}>{meta.labelLow}</span>
                    </div>
                    <span
                      className="text-[10px] font-black px-2 py-0.5 rounded-full"
                      style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
                    >{degreeLabel}</span>
                  </div>
                  <div className="relative h-3 rounded-full overflow-hidden" style={{ background: 'rgba(180,150,80,0.1)' }}>
                    <div className="absolute left-1/2 top-0 bottom-0 w-px z-10" style={{ background: 'rgba(180,150,80,0.3)' }} />
                    {isHigh
                      ? <div className="absolute top-0 bottom-0 rounded-full transition-all" style={{ right: '50%', width: `${(pct - 50) * 2}%`, background: color }} />
                      : <div className="absolute top-0 bottom-0 rounded-full transition-all" style={{ left: '50%', width: `${(50 - pct) * 2}%`, background: color }} />
                    }
                  </div>
                  <div className="flex justify-between text-[10px] mt-0.5 font-bold" style={{ color: 'rgba(180,150,80,0.35)' }}>
                    <span>{meta.labelHigh}</span>
                    <span>{Math.round(Math.abs(pct - 50) * 2)}%</span>
                    <span>{meta.labelLow}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] font-bold tracking-widest mt-5 text-right" style={{ color: 'rgba(180,150,80,0.35)' }}>✦ 性癖16タイプ分析 ✦</p>
        </div>
      </div>

      {/* 相性の良いタイプ */}
      <div className="w-full max-w-sm mb-6">
        <p className="text-[11px] font-black tracking-[0.3em] uppercase mb-4 text-center" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ 相性の良いタイプ ✦</p>
        <div className="space-y-3">
          {COMPATIBILITY[typeKey].map((compatKey, index) => {
            const compatType = QUIZ_TYPES[compatKey];
            const rankLabel = COMPATIBILITY_LABELS[index];
            const rankColors = ['#FF6B6B', '#FDCB6E', '#74B9FF'];
            const rankColor = rankColors[index];
            return (
              <Link
                key={compatKey}
                href={`/quiz/result/${compatKey}`}
                className="flex items-center gap-4 rounded-2xl p-4 active:scale-[0.98] transition-transform"
                style={{
                  background: 'rgba(28,24,18,0.9)',
                  border: `1px solid ${compatType.color}40`,
                  boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
                }}
              >
                <Image
                  src={`/quiz/${compatKey}.png`}
                  alt={compatType.name}
                  width={56}
                  height={56}
                  className="rounded-xl object-contain flex-shrink-0"
                  style={{ background: `${compatType.color}22` }}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span
                      className="text-[9px] font-black px-2 py-0.5 rounded-full flex-shrink-0"
                      style={{ background: `${rankColor}22`, color: rankColor, border: `1px solid ${rankColor}55` }}
                    >
                      {rankLabel}
                    </span>
                    <span className="text-[10px] font-black tracking-widest" style={{ color: 'rgba(180,150,80,0.4)' }}>
                      {compatKey.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-[15px] font-black leading-tight truncate" style={{ color: '#f0e6d3' }}>
                    {compatType.emoji} {compatType.name}
                  </p>
                  <p className="text-[11px] mt-0.5 truncate" style={{ color: compatType.color }}>{compatType.tagline}</p>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* シェアボタン */}
      <div className="w-full max-w-sm space-y-3 mb-6">
        <SaveImageButton cardRef={shareCardRef} typeKey={typeKey} quizType={quizType} charDataUrl={charDataUrl} onCharDataUrlReady={setCharDataUrl} />
        <button
          onClick={shareToX}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: 'rgba(255,255,255,0.08)', boxShadow: '0 4px 0 rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.2)', color: '#f0e6d3' }}
        >
          <span className="text-lg">𝕏</span> ポストして友だちに教える
        </button>
        <button
          onClick={shareToLine}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#06C755', boxShadow: '0 4px 0 #04a344', border: '1px solid #08e060' }}
        >
          <span className="text-lg">💬</span> LINEで送る
        </button>
        <button
          onClick={copyLink}
          className="w-full rounded-2xl py-4 font-black flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: 'rgba(180,150,80,0.1)', border: '1px solid rgba(180,150,80,0.35)', color: '#e8d5a0' }}
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
            target="_blank"
            rel="noopener noreferrer"
            className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] active:translate-y-[1px] transition-transform"
            style={{ background: 'linear-gradient(90deg, #7b2d8b, #c4337a)', boxShadow: '0 4px 0 #5a1a6b', border: '2px solid #9b3dab' }}
            onClick={() => trackEvent('quiz_swipe_cta_click', { type: typeKey })}
          >
            性癖ラボを試してみる ✦
          </Link>
        </div>
      )}

      <button onClick={() => router.push('/quiz')} className="text-sm font-bold underline underline-offset-4" style={{ color: 'rgba(180,150,80,0.5)' }}>
        もう一度診断する
      </button>
    </div>
  );
}

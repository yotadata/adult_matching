'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import QRCode from 'qrcode';
import { QUIZ_TYPES, AXIS_META, COMPATIBILITY, COMPATIBILITY_LABELS, QuizTypeKey, QuizType, Axis } from '../../data';
import { trackEvent } from '@/lib/analytics';

const CTA_SHOWN_KEY = 'quiz_male_cta_shown';
const AXES: Axis[] = ['ds', 'pe', 'nx', 'cw'];

const JP_FONT = '"Hiragino Kaku Gothic ProN","Hiragino Sans","Yu Gothic","Noto Sans JP",sans-serif';

// ─── Canvas で直接シェア画像を生成（縦長・下詰め） ───────────────────────────
async function buildShareImage(params: {
  typeKey: QuizTypeKey;
  quizType: QuizType;
  axes: { axis: Axis; pct: number }[];
  charDataUrl: string;
  qrDataUrl: string;
}): Promise<string> {
  const { typeKey, quizType, axes, charDataUrl, qrDataUrl } = params;
  const W = 750;
  const tx = 36; // 横余白

  // ── テキストエリアの高さを先に計算 ──
  const offCtx = document.createElement('canvas').getContext('2d')!;
  offCtx.font = `400 15px ${JP_FONT}`;
  let textH = 32; // 上パディング
  textH += 58;    // タイトル (52px)
  textH += 8;     // gap
  textH += 22;    // タグライン (17px)
  textH += 16;    // gap
  // 説明文の行数を計算
  let line = '';
  let descLines = 1;
  for (const ch of quizType.description.split('')) {
    const test = line + ch;
    if (offCtx.measureText(test).width > W - tx * 2 && line) { line = ch; descLines++; }
    else line = test;
  }
  textH += descLines * 22; // 説明文 (15px × 22lineH)
  textH += 24;  // gap + 区切り線
  textH += 20;  // 区切り線後gap
  textH += axes.length * 36; // 軸バー
  textH += 36;  // フッター + 下パディング
  const H = textH + 600; // 残り全部がキャラエリア（最低600px）
  const charH = H - textH;

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d')!;
  await document.fonts.ready;

  // 背景
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#0d0b08'); bg.addColorStop(1, '#1a1510');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

  // キャラエリア背景
  ctx.fillStyle = quizType.color + '38'; ctx.fillRect(0, 0, W, charH);
  const rg = ctx.createRadialGradient(W / 2, charH / 2, 0, W / 2, charH / 2, W * 0.6);
  rg.addColorStop(0, quizType.color + '30'); rg.addColorStop(1, 'transparent');
  ctx.fillStyle = rg; ctx.fillRect(0, 0, W, charH);

  // キャラ画像
  if (charDataUrl) {
    await new Promise<void>((resolve) => {
      const img = new window.Image();
      img.onload = () => {
        const size = Math.min(charH - 60, 480);
        const x = (W - size) / 2, y = (charH - size) / 2 + 10;
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.9)'; ctx.shadowBlur = 40; ctx.shadowOffsetY = 16;
        ctx.drawImage(img, x, y, size, size);
        ctx.restore(); resolve();
      };
      img.onerror = () => resolve();
      img.src = charDataUrl;
    });
  }

  // キャラエリア下フェード
  const fade = ctx.createLinearGradient(0, charH - 120, 0, charH);
  fade.addColorStop(0, 'transparent'); fade.addColorStop(1, '#0d0b08');
  ctx.fillStyle = fade; ctx.fillRect(0, charH - 120, W, 120);

  // ヘッダーラベル
  ctx.font = `600 22px ${JP_FONT}`; ctx.fillStyle = 'rgba(180,150,80,0.65)';
  ctx.fillText('偏愛16診断', 28, 44);

  // タイプキーバッジ
  const letters = typeKey.toUpperCase().split('');
  let bx = W - 24;
  for (let i = letters.length - 1; i >= 0; i--) {
    const bw = 42; bx -= bw + 6;
    ctx.fillStyle = quizType.color + '30'; ctx.strokeStyle = quizType.color + '66'; ctx.lineWidth = 1.5;
    roundRect(ctx, bx, 20, bw, 28, 14); ctx.fill(); ctx.stroke();
    ctx.font = `900 15px ${JP_FONT}`; ctx.fillStyle = quizType.color;
    ctx.textAlign = 'center'; ctx.fillText(letters[i], bx + bw / 2, 39);
  }
  ctx.textAlign = 'left';

  // QRコード + ドメイン（右下に大きめに配置）
  if (qrDataUrl) {
    await new Promise<void>((resolve) => {
      const qr = new window.Image();
      qr.onload = () => {
        ctx.fillStyle = 'white';
        roundRect(ctx, W - 116, charH - 122, 100, 100, 10); ctx.fill();
        ctx.drawImage(qr, W - 111, charH - 117, 90, 90); resolve();
      };
      qr.onerror = () => resolve(); qr.src = qrDataUrl;
    });
  }

  // ── テキストエリア（下詰め）──
  let ty = charH + 32;

  ctx.font = `900 52px ${JP_FONT}`; ctx.fillStyle = '#f0e6d3';
  ctx.fillText(quizType.name, tx, ty + 52); ty += 58 + 8;

  ctx.font = `700 17px ${JP_FONT}`; ctx.fillStyle = quizType.color;
  ctx.fillText(quizType.tagline, tx, ty + 17); ty += 22 + 16;

  ctx.font = `400 15px ${JP_FONT}`; ctx.fillStyle = 'rgba(200,180,140,0.65)';
  ty = wrapText(ctx, quizType.description, tx, ty, W - tx * 2, 22) + 24;

  // 区切り線
  ctx.strokeStyle = 'rgba(180,150,80,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(W - tx, ty); ctx.stroke();
  ty += 20;

  // 軸バー
  for (const { axis, pct } of axes) {
    const meta = AXIS_META[axis];
    const isHigh = pct >= 50;
    const color = isHigh ? meta.colorHigh : meta.colorLow;
    let degreeLabel: string;
    if (pct >= 80)      degreeLabel = meta.degreesHigh[0];
    else if (pct >= 60) degreeLabel = meta.degreesHigh[1];
    else if (pct >= 40) degreeLabel = 'ニュートラル';
    else if (pct >= 20) degreeLabel = meta.degreesLow[1];
    else                degreeLabel = meta.degreesLow[0];

    // 左: labelHigh  中: バー  右: labelLow  度合いバッジ
    const axisLabelW = 52, degreeW = 88;
    const barX = tx + axisLabelW + 8, barW = W - tx * 2 - axisLabelW * 2 - degreeW - 24;

    ctx.font = `700 14px ${JP_FONT}`;
    ctx.textAlign = 'right';
    ctx.fillStyle = isHigh ? color : 'rgba(200,180,140,0.35)';
    ctx.fillText(meta.labelHigh, tx + axisLabelW, ty + 9);

    ctx.textAlign = 'left';
    ctx.fillStyle = !isHigh ? color : 'rgba(200,180,140,0.35)';
    ctx.fillText(meta.labelLow, barX + barW + 8, ty + 9);

    ctx.fillStyle = 'rgba(180,150,80,0.1)';
    roundRect(ctx, barX, ty, barW, 10, 5); ctx.fill();
    ctx.fillStyle = 'rgba(180,150,80,0.3)'; ctx.fillRect(barX + barW / 2, ty, 1.5, 10);

    const fillW = Math.abs(pct - 50) / 50 * (barW / 2);
    ctx.fillStyle = color;
    roundRect(ctx, isHigh ? barX + barW / 2 - fillW : barX + barW / 2, ty, fillW, 10, 5); ctx.fill();

    // 度合いバッジ（右端）
    const badgeX = tx + axisLabelW + 8 + barW + axisLabelW + 16;
    const badgeH = 20, badgeY = ty - 5;
    ctx.font = `700 12px ${JP_FONT}`;
    ctx.fillStyle = color + '28';
    roundRect(ctx, badgeX, badgeY, degreeW, badgeH, 10); ctx.fill();
    ctx.strokeStyle = color + '66'; ctx.lineWidth = 1;
    roundRect(ctx, badgeX, badgeY, degreeW, badgeH, 10); ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = 'center'; ctx.fillText(degreeLabel, badgeX + degreeW / 2, badgeY + 13);

    ctx.textAlign = 'left'; ty += 36;
  }

  // フッター
  ctx.font = `600 15px ${JP_FONT}`; ctx.fillStyle = 'rgba(180,150,80,0.3)';
  ctx.textAlign = 'right'; ctx.fillText('✦ 偏愛16診断 ✦', W - tx, ty + 16);
  ctx.textAlign = 'left';

  return canvas.toDataURL('image/png');
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function wrapText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, lineH: number): number {
  let line = '';
  let cy = y;
  for (const ch of text.split('')) {
    const test = line + ch;
    if (ctx.measureText(test).width > maxW && line) {
      ctx.fillText(line, x, cy);
      line = ch;
      cy += lineH;
    } else {
      line = test;
    }
  }
  if (line) ctx.fillText(line, x, cy);
  return cy;
}

// ─── 「画像で保存/シェア」ボタン ────────────────────────────────────────────
function SaveImageButton({
  typeKey,
  quizType,
  axes,
  charDataUrl,
  qrDataUrl,
  onCharDataUrlReady,
}: {
  typeKey: QuizTypeKey;
  quizType: QuizType;
  axes: { axis: Axis; pct: number }[];
  charDataUrl: string;
  qrDataUrl: string;
  onCharDataUrlReady: (url: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleSave = async () => {
    if (loading) return;
    setLoading(true);
    setErrorMsg('');
    trackEvent('quiz_save_image', { type: typeKey });
    try {
      // データURLを確保（未取得なら今すぐfetch）
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

      const capturedDataUrl = await buildShareImage({ typeKey, quizType, axes, charDataUrl: resolvedCharDataUrl, qrDataUrl });

      const blob = await (await fetch(capturedDataUrl)).blob();
      const file = new File([blob], `seiheki_${typeKey}.png`, { type: 'image/png' });

      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({
          files: [file],
          title: `私の偏愛16診断タイプは「${quizType.name}」でした！`,
          text: quizType.tagline,
        });
        trackEvent('quiz_image_shared', { type: typeKey, method: 'webshare' });
      } else {
        const newTab = window.open();
        if (newTab) {
          newTab.document.write(`<img src="${capturedDataUrl}" style="max-width:100%;display:block;" />`);
          newTab.document.close();
        } else {
          const a = document.createElement('a');
          a.href = capturedDataUrl;
          a.download = `seiheki_${typeKey}.png`;
          a.click();
        }
        trackEvent('quiz_image_shared', { type: typeKey, method: 'download' });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('画像生成失敗:', e);
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {errorMsg && (
        <p className="text-[11px] text-red-400 mb-2 px-1 break-all">{errorMsg}</p>
      )}
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
    </>
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
  const [showCTA, setShowCTA] = useState(false);
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [charDataUrl, setCharDataUrl] = useState('');
  const [showDetail, setShowDetail] = useState(false);
  const shareVersion = '3';

  useEffect(() => {
    QRCode.toDataURL('https://www.seihekilab.com/quiz', { width: 128, margin: 1, color: { dark: '#000000', light: '#ffffff' } })
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

  useEffect(() => {
    if (searchParams.get('v')) return;
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.set('v', shareVersion);
    router.replace(`/quiz/result/${typeKey}?${nextParams.toString()}`);
  }, [router, searchParams, shareVersion, typeKey]);

  const rawScores = (() => {
    try { return JSON.parse(decodeURIComponent(searchParams.get('scores') ?? '{}')); }
    catch { return {}; }
  })();

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['senc'];
  const scoresParam = searchParams.get('scores') ?? '';
  const versionParam = searchParams.get('v') ?? shareVersion;
  const shareUrl = scoresParam
    ? `https://www.seihekilab.com/quiz/result/${typeKey}?scores=${encodeURIComponent(scoresParam)}&v=${encodeURIComponent(versionParam)}`
    : `https://www.seihekilab.com/quiz/result/${typeKey}?v=${encodeURIComponent(versionParam)}`;
  const axes: { axis: Axis; pct: number }[] = AXES.map((axis) => ({
    axis,
    pct: typeof rawScores[axis] === 'number' ? rawScores[axis] : 50,
  }));

  const axisLine = axes.map(({ axis, pct }) => {
    const meta = AXIS_META[axis];
    return pct >= 50 ? meta.labelHigh : meta.labelLow;
  }).join('｜');
  const shareText = `偏愛16診断やってみた🔍\n私のタイプは「${quizType.name}」${quizType.emoji}\n${axisLine}\n\n#偏愛16診断\n${shareUrl}`;

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
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`,
      '_blank', 'noopener'
    );
  };
  const shareToLine = () => {
    trackEvent('quiz_share', { method: 'line', type: typeKey });
    window.open(`https://line.me/R/msg/text/?${encodeURIComponent(shareText)}`, '_blank', 'noopener');
  };
  const copyLink = async () => {
    trackEvent('quiz_share', { method: 'copy_link', type: typeKey });
    await navigator.clipboard.writeText(shareText);
  };

  return (
    <div className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8">
      {showCTA && <MaleCTAModal typeKey={typeKey} onClose={closeCTA} />}

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

          {/* アコーディオン: もっと詳しく */}
          <div className="mt-5" style={{ borderTop: '1px solid rgba(180,150,80,0.2)' }}>
            <button
              onClick={() => setShowDetail(v => !v)}
              className="w-full flex items-center justify-between pt-4 pb-2"
            >
              <span className="text-[11px] font-black tracking-widest uppercase" style={{ color: 'rgba(180,150,80,0.6)' }}>
                ✦ もっと詳しく見る ✦
              </span>
              <span className="text-[13px] transition-transform duration-200" style={{ color: 'rgba(180,150,80,0.5)', transform: showDetail ? 'rotate(180deg)' : 'rotate(0deg)', display: 'inline-block' }}>▼</span>
            </button>

            {showDetail && (
              <div className="space-y-5 pb-2">
                <p className="text-sm leading-relaxed" style={{ color: 'rgba(200,180,140,0.75)' }}>
                  {quizType.detailDescription}
                </p>

                <div>
                  <p className="text-[10px] font-black tracking-widest uppercase mb-2" style={{ color: 'rgba(180,150,80,0.5)' }}>好きなプレイ</p>
                  <div className="flex flex-wrap gap-2">
                    {quizType.favPlay.map((play, i) => (
                      <span
                        key={i}
                        className="text-[11px] font-bold px-2.5 py-1 rounded-full"
                        style={{ background: `${quizType.color}18`, color: quizType.color, border: `1px solid ${quizType.color}45` }}
                      >
                        {play}
                      </span>
                    ))}
                  </div>
                </div>

                <div
                  className="rounded-2xl p-4"
                  style={{ background: `${quizType.color}12`, border: `1px solid ${quizType.color}30` }}
                >
                  <p className="text-[10px] font-black tracking-widest uppercase mb-2" style={{ color: 'rgba(180,150,80,0.5)' }}>あるある</p>
                  <p className="text-sm leading-relaxed" style={{ color: 'rgba(200,180,140,0.8)' }}>
                    {quizType.trivia}
                  </p>
                </div>
              </div>
            )}
          </div>

          <p className="text-[10px] font-bold tracking-widest mt-5 text-right" style={{ color: 'rgba(180,150,80,0.35)' }}>✦ 偏愛16診断 ✦</p>
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
        <SaveImageButton typeKey={typeKey} quizType={quizType} axes={axes} charDataUrl={charDataUrl} qrDataUrl={qrDataUrl} onCharDataUrlReady={setCharDataUrl} />
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

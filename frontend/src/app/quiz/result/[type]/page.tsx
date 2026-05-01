'use client';

import { useSearchParams } from 'next/navigation';
import { useRouter } from 'next/navigation';
import { Suspense } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { QUIZ_TYPES, AXIS_META, QuizTypeKey, Axis } from '../../data';

export default async function ResultPage({ params }: { params: Promise<{ type: string }> }) {
  const { type } = await params;
  return (
    <Suspense fallback={
      <div className="min-h-[calc(100vh-56px)] flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, #ffecd2, #ffd1dc)' }} />
    }>
      <ResultContent typeKey={type as QuizTypeKey} />
    </Suspense>
  );
}

function ResultContent({ typeKey }: { typeKey: QuizTypeKey }) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const gender = searchParams.get('gender') ?? 'other';
  const isMale = gender === 'male';

  // scores パラメータ（URLから）: { ds: 73, nx: 52, pe: 87, hl: 71 }
  const rawScores = (() => {
    try { return JSON.parse(decodeURIComponent(searchParams.get('scores') ?? '{}')); }
    catch { return {}; }
  })();

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['sneh'];
  const shareText = `私の性癖16タイプは「${quizType.name}」でした！${quizType.tagline}\n\nあなたは？👇`;
  const scoresParam = searchParams.get('scores') ?? '';
  const shareUrl = `https://seihekilab.com/quiz/result/${typeKey}?gender=${gender}&scores=${scoresParam}`;

  const axes: { axis: Axis; pct: number }[] = (['ds', 'nx', 'pe', 'hl'] as Axis[]).map((axis) => ({
    axis,
    pct: typeof rawScores[axis] === 'number' ? rawScores[axis] : 50,
  }));

  const shareToX = () => {
    window.open(
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}&hashtags=性癖16タイプ診断`,
      '_blank', 'noopener'
    );
  };
  const shareToLine = () => {
    window.open(`https://line.me/R/msg/text/?${encodeURIComponent(`${shareText}\n${shareUrl}`)}`, '_blank', 'noopener');
  };
  const copyLink = async () => { await navigator.clipboard.writeText(shareUrl); };

  return (
    <div
      className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8"
      style={{ background: `linear-gradient(135deg, ${quizType.color}22 0%, #ffd1dc22 100%)` }}
    >
      <p className="text-xs font-bold tracking-[0.3em] text-gray-400 uppercase mb-4">診断結果</p>

      {/* メインカード */}
      <div
        className="w-full max-w-sm rounded-3xl overflow-hidden mb-5"
        style={{
          background: '#fffdf8',
          boxShadow: `0 2px 0 ${quizType.accent}55, 0 5px 0 ${quizType.accent}88, 0 10px 30px rgba(0,0,0,0.10)`,
        }}
      >
        {/* カラーヘッダー */}
        <div className="relative h-64 flex items-center justify-center" style={{ background: quizType.color }}>
          <div className="absolute inset-0 rounded-t-3xl" style={{ background: 'rgba(255,255,255,0.25)' }} />
          <div className="absolute inset-2 rounded-2xl" style={{ background: `${quizType.accent}10`, boxShadow: `inset 0 2px 8px ${quizType.accent}22` }} />
          <div className="absolute bottom-0 left-0 right-0 h-8" style={{ background: '#fffdf8', clipPath: 'ellipse(60% 100% at 50% 100%)' }} />
          <Image
            src={`/quiz/${typeKey}.png`}
            alt={quizType.name}
            width={260}
            height={260}
            className="relative object-contain"
            style={{ filter: 'drop-shadow(0 6px 16px rgba(0,0,0,0.2))' }}
          />
        </div>

        <div className="px-6 pt-4 pb-6">
          {/* タイプキー */}
          <div className="flex items-center gap-1.5 mb-3">
            {typeKey.toUpperCase().split('').map((c, i) => (
              <span key={i} className="text-[10px] font-black px-2 py-0.5 rounded-full" style={{ background: quizType.color, color: quizType.accent }}>{c}</span>
            ))}
          </div>

          <h2 className="text-[28px] font-black text-gray-800 leading-tight mb-1">{quizType.name}</h2>
          <p className="text-sm font-bold mb-4" style={{ color: quizType.accent }}>{quizType.tagline}</p>
          <p className="text-sm leading-relaxed text-gray-600 mb-6">{quizType.description}</p>

          {/* 軸スコアバー */}
          <div className="space-y-4 border-t border-gray-100 pt-5">
            <p className="text-[10px] font-black tracking-widest text-gray-400 uppercase">あなたの傾向</p>
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
                      <span className="text-[11px] font-black text-gray-500">{meta.labelHigh}</span>
                      <span className="text-[10px] text-gray-300">⇄</span>
                      <span className="text-[11px] font-black text-gray-500">{meta.labelLow}</span>
                    </div>
                    <span className="text-[11px] font-black" style={{ color }}>{degreeLabel}</span>
                  </div>
                  {/* バー */}
                  <div className="relative h-3 rounded-full bg-gray-100 overflow-hidden">
                    {/* 中央線 */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 z-10" />
                    <div
                      className="absolute left-0 top-0 bottom-0 rounded-full transition-all"
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-gray-400 mt-0.5 font-bold">
                    <span>{meta.labelHigh}</span>
                    <span>{pct}%</span>
                    <span>{meta.labelLow}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] text-gray-200 font-bold tracking-widest mt-5 text-right">性癖16タイプ診断</p>
        </div>
      </div>

      {/* シェアボタン */}
      <div className="w-full max-w-sm space-y-3 mb-5">
        <button onClick={shareToX} className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform" style={{ background: '#000', boxShadow: '0 4px 0 #333' }}>
          <span className="text-lg">𝕏</span> ポストして友だちに教える
        </button>
        <button onClick={shareToLine} className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform" style={{ background: '#06C755', boxShadow: '0 4px 0 #04a344' }}>
          <span className="text-lg">💬</span> LINEで送る
        </button>
        <button onClick={copyLink} className="w-full rounded-2xl py-4 font-black flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform" style={{ background: '#fff8f0', border: '2.5px solid #ffb347', boxShadow: '0 3px 0 #e08020', color: '#3d1a00' }}>
          🔗 リンクをコピー
        </button>
      </div>

      {/* 男性向けCTA */}
      {isMale && (
        <div className="w-full max-w-sm rounded-3xl p-6 mb-5" style={{ background: 'linear-gradient(135deg, #1a0d2e, #2a1020)', boxShadow: '0 4px 0 #0d0616, 0 8px 24px rgba(0,0,0,0.3)' }}>
          <p className="text-xs font-bold tracking-widest text-purple-300/70 uppercase mb-2">あなたへのおすすめ</p>
          <p className="text-xl font-black text-white mb-2">あなたの性癖に合う動画、見つかるかも</p>
          <p className="text-sm text-white/60 mb-4">AIがあなたの好みを学習して、刺さる動画だけをおすすめ。スワイプして探してみよう。</p>
          <Link href="/" className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px]" style={{ background: 'linear-gradient(90deg, #9333ea, #ec4899)', boxShadow: '0 4px 0 #6b21a8' }}>
            性癖ラボを試してみる 🔥
          </Link>
        </div>
      )}

      <button onClick={() => router.push('/quiz')} className="text-sm font-bold text-gray-400 underline underline-offset-4">
        もう一度診断する
      </button>
    </div>
  );
}

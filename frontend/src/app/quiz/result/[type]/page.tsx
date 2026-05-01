'use client';

import { useSearchParams } from 'next/navigation';
import { useRouter } from 'next/navigation';
import { Suspense, useRef } from 'react';
import Link from 'next/link';
import { QUIZ_TYPES, QuizTypeKey } from '../../data';

export default async function ResultPage({ params }: { params: Promise<{ type: string }> }) {
  const { type } = await params;
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 50%, #ffd1dc 100%)' }} />}>
      <ResultContent typeKey={type as QuizTypeKey} />
    </Suspense>
  );
}

function ResultContent({ typeKey }: { typeKey: QuizTypeKey }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const cardRef = useRef<HTMLDivElement>(null);
  const gender = searchParams.get('gender') ?? 'other';
  const isMale = gender === 'male';

  const quizType = QUIZ_TYPES[typeKey] ?? QUIZ_TYPES['sneh'];
  const shareText = `私の性癖パーソナリティは「${quizType.name}」でした！${quizType.tagline}\n\nあなたは？👇`;
  const shareUrl = `https://seihekilab.com/quiz/result/${typeKey}`;

  const shareToX = () => {
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}&hashtags=性癖パーソナリティ診断`;
    window.open(url, '_blank', 'noopener');
  };

  const shareToLine = () => {
    const url = `https://line.me/R/msg/text/?${encodeURIComponent(`${shareText}\n${shareUrl}`)}`;
    window.open(url, '_blank', 'noopener');
  };

  const copyLink = async () => {
    await navigator.clipboard.writeText(shareUrl);
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12"
      style={{ background: `linear-gradient(135deg, ${quizType.color}33 0%, ${quizType.color}22 50%, #ffd1dc33 100%)` }}
    >
      <p className="text-xs font-bold tracking-[0.3em] text-gray-500 uppercase mb-4">診断結果</p>

      {/* 結果カード */}
      <div
        ref={cardRef}
        className="w-full max-w-sm rounded-3xl overflow-hidden mb-6"
        style={{
          background: '#fffdf8',
          boxShadow: `0 2px 0 ${quizType.accent}55, 0 5px 0 ${quizType.accent}88, 0 8px 0 ${quizType.accent}aa, 0 12px 30px rgba(0,0,0,0.12)`,
        }}
      >
        {/* カラーヘッダー（フェルト切り絵的な重なり表現） */}
        <div
          className="relative h-48 flex items-center justify-center"
          style={{ background: quizType.color }}
        >
          {/* 波形の切り抜き */}
          <div
            className="absolute bottom-0 left-0 right-0 h-10"
            style={{
              background: '#fffdf8',
              clipPath: 'ellipse(60% 100% at 50% 100%)',
            }}
          />
          {/* フェルト風の内側シャドウ層 */}
          <div
            className="absolute inset-2 rounded-2xl"
            style={{
              background: `${quizType.accent}22`,
              boxShadow: `inset 0 2px 8px ${quizType.accent}44`,
            }}
          />
          <span className="relative text-7xl drop-shadow-lg">{quizType.emoji}</span>
        </div>

        <div className="px-7 pt-5 pb-7">
          {/* タイプキー */}
          <div className="flex items-center gap-2 mb-3">
            {typeKey.toUpperCase().split('').map((c, i) => (
              <span
                key={i}
                className="text-xs font-black px-2 py-0.5 rounded-full"
                style={{ background: quizType.color, color: quizType.accent }}
              >
                {c}
              </span>
            ))}
          </div>

          {/* タイプ名 */}
          <h2 className="text-3xl font-black text-gray-800 mb-1 leading-tight">
            {quizType.name}
          </h2>
          <p className="text-sm font-bold mb-4" style={{ color: quizType.accent }}>
            {quizType.tagline}
          </p>

          {/* 説明文 */}
          <p className="text-sm leading-relaxed text-gray-600">
            {quizType.description}
          </p>

          {/* 透かしロゴ */}
          <p className="text-[10px] text-gray-300 font-bold tracking-widest mt-5 text-right">
            性癖パーソナリティ診断
          </p>
        </div>
      </div>

      {/* シェアボタン */}
      <div className="w-full max-w-sm space-y-3 mb-6">
        <button
          onClick={shareToX}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#000', boxShadow: '0 4px 0 #333' }}
        >
          <span className="text-lg">𝕏</span> ポストして友だちに教える
        </button>
        <button
          onClick={shareToLine}
          className="w-full rounded-2xl py-4 font-black text-white flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#06C755', boxShadow: '0 4px 0 #04a344' }}
        >
          <span className="text-lg">💬</span> LINEで送る
        </button>
        <button
          onClick={copyLink}
          className="w-full rounded-2xl py-4 font-black flex items-center justify-center gap-2 text-[15px] active:translate-y-[1px] transition-transform"
          style={{ background: '#fff8f0', border: '2.5px solid #ffb347', boxShadow: '0 3px 0 #e08020', color: '#3d1a00' }}
        >
          🔗 リンクをコピー
        </button>
      </div>

      {/* 男性向けCTA */}
      {isMale && (
        <div
          className="w-full max-w-sm rounded-3xl p-6 mb-6"
          style={{
            background: 'linear-gradient(135deg, #1a0d2e, #2a1020)',
            boxShadow: '0 4px 0 #0d0616, 0 8px 24px rgba(0,0,0,0.3)',
          }}
        >
          <p className="text-xs font-bold tracking-widest text-purple-300/70 uppercase mb-2">あなたへのおすすめ</p>
          <p className="text-xl font-black text-white mb-2">
            あなたの性癖に合う動画、見つかるかも
          </p>
          <p className="text-sm text-white/60 mb-4">
            AIがあなたの好みを学習して、刺さる動画だけをおすすめ。スワイプして探してみよう。
          </p>
          <Link
            href="/"
            className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px]"
            style={{ background: 'linear-gradient(90deg, #9333ea, #ec4899)', boxShadow: '0 4px 0 #6b21a8' }}
          >
            性癖ラボを試してみる 🔥
          </Link>
        </div>
      )}

      {/* もう一度やる */}
      <button
        onClick={() => router.push('/quiz')}
        className="text-sm font-bold text-gray-400 underline underline-offset-4"
      >
        もう一度診断する
      </button>
    </div>
  );
}

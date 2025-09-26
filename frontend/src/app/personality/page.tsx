'use client';

import Link from 'next/link';

export default function PersonalityPage() {

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white px-4 text-center">
      <div className="max-w-2xl">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 leading-tight">
          あなたの性癖パーソナリティを解き明かす
        </h1>
        <p className="text-base sm:text-lg md:text-xl mb-8 text-white/80">
          30の質問に答えるだけで、あなたの隠された性癖タイプがわかります。
          4つの異なる軸からあなたの傾向を分析し、16のユニークなタイプの中からあなたのパーソナリティを診断します。
        </p>
        <Link 
          href="/personality/questions"
          className="px-8 py-4 rounded-full bg-white text-gray-900 font-bold text-lg shadow-lg hover:bg-gray-200 transition-all duration-300 transform hover:scale-105"
        >
          診断を始める
        </Link>
      </div>
    </div>
  );
}

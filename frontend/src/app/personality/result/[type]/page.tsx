'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { personalityTypes } from '@/data/personalityQuiz';

export default function ResultPage() {
  const params = useParams();
  const type = params.type as string;

  const result = personalityTypes[type];

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen text-white">
        <h1 className="text-2xl font-bold mb-4">診断結果が見つかりません</h1>
        <p className="mb-8">URLが正しいか確認してください。</p>
        <Link href="/swipe" className="px-6 py-2 rounded-full bg-white text-gray-900 font-bold">
          スワイプに戻る
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white px-4 text-center">
      <div className="max-w-2xl">
        <p className="text-lg text-amber-400 font-bold">あなたの性癖タイプは...</p>
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold my-4 text-amber-300">
          {result.title}
        </h1>
        <div className="p-6 mt-4 bg-black/20 rounded-lg shadow-inner">
          <p className="text-base sm:text-lg md:text-xl text-left text-white/90">
            {result.description}
          </p>
        </div>

        <div className="mt-10 flex flex-col sm:flex-row gap-4">
          <Link 
            href="/swipe"
            className="w-full sm:w-auto px-8 py-3 rounded-full bg-white text-gray-900 font-bold shadow-lg hover:bg-gray-200 transition-all duration-300 transform hover:scale-105"
          >
            スワイプに戻る
          </Link>
          <Link 
            href="/swipe"
            className="w-full sm:w-auto px-8 py-3 rounded-full bg-white/20 backdrop-blur border border-white/40 text-white font-bold hover:bg-white/30 transition-colors duration-300"
          >
            スワイプに戻る
          </Link>
        </div>
      </div>
    </div>
  );
}

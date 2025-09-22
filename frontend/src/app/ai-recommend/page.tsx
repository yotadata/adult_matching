'use client';
import { RotateCcw } from 'lucide-react';

export default function AiRecommendPage() {
  const handleReset = () => {
    try {
      // とりあえずの枠: 実装時に選好データ/キャッシュのクリアへ差し替え
      console.log('[AI Recommend] reset clicked');
    } catch {}
  };

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 flex flex-col items-stretch justify-start gap-6">
      <section
        className="w-full max-w-none mx-0 rounded-none sm:rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-4 sm:px-8 py-6 sm:py-8 text-white"
      >
        <header className="mb-6 flex items-center justify-between gap-3">
          <div>
            <h1 className="text-lg sm:text-xl font-extrabold tracking-tight">あなたが選んだ作品からのおすすめ</h1>
            <p className="text-sm text-white/80 mt-1">最近の評価からAIがピックアップします。</p>
          </div>
          <button
            onClick={handleReset}
            aria-label="リセット"
            className="shrink-0 inline-flex items-center justify-center rounded-md bg-white/20 hover:bg-white/30 backdrop-blur p-2 text-white transition"
            title="リセット"
          >
            <RotateCcw size={18} className="text-white" />
          </button>
        </header>

        {/* プレースホルダー（とりあえず10枠）*/}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 sm:gap-4">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="aspect-[4/3] rounded-xl bg-white/30 border border-white/40 backdrop-blur-sm shadow-inner" />
          ))}
        </div>
      </section>


      {/* 直近の全体トレンド */}
      <section
        className="w-full max-w-none mx-0 rounded-none sm:rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-4 sm:px-8 py-6 sm:py-8 text-white"
      >
        <header className="mb-6">
          <h2 className="text-lg sm:text-xl font-extrabold tracking-tight">直近の全体トレンド</h2>
          <p className="text-sm text-white/80 mt-1">サイト全体の直近トレンド（プレースホルダー）。</p>
        </header>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 sm:gap-4">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="aspect-[4/3] rounded-xl bg-white/30 border border-white/40 backdrop-blur-sm shadow-inner" />
          ))}
        </div>
      </section>
    </main>
  );
}

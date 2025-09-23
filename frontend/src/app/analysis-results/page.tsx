'use client';
import { Share2 } from 'lucide-react';

export default function AnalysisResultsPage() {
  const handleShare = async () => {
    try {
      const shareData = {
        title: 'あなたの性癖分析結果',
        text: '性癖分析の結果をシェアしました。',
        url: typeof window !== 'undefined' ? window.location.href : undefined,
      } as ShareData;
      if (navigator.share) {
        await navigator.share(shareData);
        return;
      }
      if (navigator.clipboard && typeof window !== 'undefined') {
        await navigator.clipboard.writeText(window.location.href);
        window.alert('リンクをクリップボードにコピーしました');
        return;
      }
    } catch {}
  };
  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 flex items-start justify-center">
      <section
        className="w-full max-w-none mx-0 rounded-none sm:rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-4 sm:px-8 py-6 sm:py-8 text-white"
      >
        <header className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-lg sm:text-xl font-extrabold tracking-tight">あなたの性癖分析結果</h1>
          <button
            onClick={handleShare}
            aria-label="共有"
            className="shrink-0 inline-flex items-center justify-center rounded-md bg-white/20 hover:bg-white/30 backdrop-blur p-2 text-white transition"
            title="共有"
          >
            <Share2 size={18} />
          </button>
        </header>
        {/* プレースホルダー領域（分析結果が入ります） */}
        <div className="grid grid-cols-1 gap-4">
          <div className="rounded-xl bg-white/10 border border-white/30 backdrop-blur p-4">
            <p className="text-sm text-white/90">分析結果の内容がここに表示されます。（プレースホルダー）</p>
          </div>
        </div>
      </section>
    </main>
  );
}

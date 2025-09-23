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
<<<<<<< HEAD
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
        {/* 強めの白カードで3セクション */}
        <div className="grid grid-cols-1 gap-4">
          <section className="rounded-xl bg-white/70 backdrop-blur-md shadow-lg border border-white/40 p-4 text-gray-900">
            <h2 className="text-base font-bold mb-2">あなたの好きなタグ</h2>
            <div className="h-24 rounded-md bg-white/40"></div>
          </section>
          <section className="rounded-xl bg-white/70 backdrop-blur-md shadow-lg border border-white/40 p-4 text-gray-900">
            <h2 className="text-base font-bold mb-2">あなたの好きな女優</h2>
            <div className="h-24 rounded-md bg-white/40"></div>
          </section>
          <section className="rounded-xl bg-white/70 backdrop-blur-md shadow-lg border border-white/40 p-4 text-gray-900">
            <h2 className="text-base font-bold mb-2">LIKE比率</h2>
            <div className="h-24 rounded-md bg-white/40"></div>
          </section>
        </div>
      </section>
=======
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">性癖分析結果</h1>
      <p className="text-gray-700">ここに性癖分析の結果が表示されます。</p>
>>>>>>> ea197fe (feat: 田中リファクタリング (LFS対応版))
    </main>
  );
}

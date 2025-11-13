import { forwardRef } from 'react';
import type { AnalysisPerformer, AnalysisSummary, AnalysisTag } from '@/hooks/useAnalysisResults';

interface AnalysisShareCardProps {
  summary: AnalysisSummary | null;
  topTags: AnalysisTag[];
  topPerformers: AnalysisPerformer[];
  windowLabel: string;
}

const formatCount = (value: number | undefined | null) => {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('ja-JP');
};

const formatPercent = (value: number | null | undefined) => {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(1)}%`;
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '—';
  return parsed.toLocaleDateString('ja-JP');
};

const AnalysisShareCard = forwardRef<HTMLDivElement, AnalysisShareCardProps>(function AnalysisShareCard(
  { summary, topTags, topPerformers, windowLabel },
  ref,
) {
  const tags = topTags.slice(0, 3);
  const performers = topPerformers.slice(0, 3);
  const periodLabel = windowLabel || '全期間';

  return (
    <div
      ref={ref}
      id="analysis-share-card"
      className="w-[960px] h-[540px] rounded-[32px] text-white p-10 flex flex-col gap-6 shadow-2xl border border-white/10"
      style={{
        backgroundColor: '#050113',
        backgroundImage: 'linear-gradient(135deg, #7f1d1d 0%, #111827 55%, #020617 100%)',
        fontFamily: '"Noto Sans JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <header className="flex items-start justify-between gap-6">
        <div>
          <p className="text-sm font-semibold text-rose-200">あなたの性癖 - 嗜好分析</p>
          <h1 className="text-4xl font-extrabold tracking-tight mt-2">MY PREFERENCE REPORT</h1>
        </div>
        <div className="text-right">
          <p className="text-xs text-white/70">集計期間</p>
          <p className="text-lg font-semibold">{periodLabel}</p>
        </div>
      </header>

      <section className="grid grid-cols-3 gap-4">
        <div className="rounded-2xl bg-white/10 border border-white/20 p-4">
          <p className="text-xs uppercase tracking-wider text-white/60">LIKE 総数</p>
          <p className="text-3xl font-extrabold mt-2">{formatCount(summary?.total_likes)}</p>
          <p className="text-xs text-white/60 mt-1">サンプル {formatCount(summary?.sample_size)} 件</p>
        </div>
        <div className="rounded-2xl bg-white/10 border border-white/20 p-4">
          <p className="text-xs uppercase tracking-wider text-white/60">NOPE 総数</p>
          <p className="text-3xl font-extrabold mt-2">{formatCount(summary?.total_nope)}</p>
          <p className="text-xs text-white/60 mt-1">最新 {formatDate(summary?.latest_decision_at)}</p>
        </div>
        <div className="rounded-2xl bg-gradient-to-br from-rose-500/80 to-orange-400/70 border border-rose-200/40 p-4 shadow-lg">
          <p className="text-xs uppercase tracking-wider text-white/80">LIKE 比率</p>
          <p className="text-3xl font-extrabold mt-2">{formatPercent(summary?.like_ratio)}</p>
          <p className="text-xs text-white/80 mt-1">あなたの基準値</p>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-4">
        <div className="rounded-2xl bg-white/10 border border-white/15 p-4">
          <p className="text-sm font-semibold text-white/80 mb-3">Top Tags</p>
          {tags.length === 0 ? (
            <p className="text-xs text-white/60">データがまだありません</p>
          ) : (
            <div className="flex flex-col gap-2">
              {tags.map((tag, index) => (
                <div key={tag.tag_id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white/40 w-5 text-right">{index + 1}</span>
                    <span className="font-semibold">{tag.tag_name}</span>
                  </div>
                  <div className="text-right text-xs text-white/70">
                    <p>LIKE比 {formatPercent(tag.like_ratio)}</p>
                    <p>シェア {formatPercent(tag.share)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-2xl bg-white/10 border border-white/15 p-4">
          <p className="text-sm font-semibold text-white/80 mb-3">Top Performers</p>
          {performers.length === 0 ? (
            <p className="text-xs text-white/60">データがまだありません</p>
          ) : (
            <div className="flex flex-col gap-2">
              {performers.map((performer, index) => (
                <div key={performer.performer_id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white/40 w-5 text-right">{index + 1}</span>
                    <span className="font-semibold">{performer.performer_name}</span>
                  </div>
                  <div className="text-right text-xs text-white/70">
                    <p>LIKE比 {formatPercent(performer.like_ratio)}</p>
                    <p>シェア {formatPercent(performer.share)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <footer className="flex items-center justify-between text-xs text-white/60">
        <p>Powered by seiheki.me</p>
        <p>#あなたの性癖 を診断してシェア</p>
      </footer>
    </div>
  );
});

export default AnalysisShareCard;

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

const buildHighlightCopy = (tag?: AnalysisTag, performer?: AnalysisPerformer) => {
  if (tag && performer) {
    return `#${tag.tag_name} の作品で ${performer.performer_name} が登場すると即反応する傾向あり。沼りポイントを友だちにも共有しよう。`;
  }
  if (tag) {
    return `#${tag.tag_name} 系のキーワードに触れると指が勝手に「気になる」。次の抜けるタグを発見しよう。`;
  }
  if (performer) {
    return `${performer.performer_name} の出演作は高確率で「気になる」。ほかの人にも抜ける出演者を共有しませんか？`;
  }
  return '「気になる」の履歴からあなたのツボを解析中。結果が出たらシェアして盛り上がろう！';
};

const AnalysisShareCard = forwardRef<HTMLDivElement, AnalysisShareCardProps>(function AnalysisShareCard(
  { summary, topTags, topPerformers, windowLabel },
  ref,
) {
  const tags = topTags.slice(0, 4);
  const performers = topPerformers.slice(0, 4);
  const periodLabel = windowLabel || '全期間';
  const highlightTag = tags[0];
  const highlightPerformer = performers[0];
  const highlightCopy = buildHighlightCopy(highlightTag, highlightPerformer);

  return (
    <div
      ref={ref}
      id="analysis-share-card"
      className="w-[960px] h-[540px] rounded-[36px] text-white p-10 flex flex-col gap-6 shadow-[0_20px_60px_rgba(0,0,0,0.45)] border border-white/15"
      style={{
        backgroundColor: '#0b1120',
        backgroundImage: 'radial-gradient(circle at 20% 20%, rgba(244,114,182,0.25), transparent 45%), radial-gradient(circle at 80% 0%, rgba(129,140,248,0.25), transparent 40%), linear-gradient(135deg, #0f172a 0%, #111827 60%, #0a0f1c 100%)',
        fontFamily: '"Noto Sans JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <header className="flex items-start justify-between gap-6">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.4em] text-rose-200/80">Seiheki Profile</p>
          <h1 className="text-4xl font-black tracking-tight">あなたの性癖</h1>
          <p className="text-sm text-white/75">「気になる」履歴から見えた「今のツボ」</p>
        </div>
        <div className="text-right text-xs text-white/70">
          <p className="uppercase tracking-[0.3em] text-white/50">window</p>
          <p className="text-xl font-semibold text-white">{periodLabel}</p>
          <p className="mt-1">対象「気になる」 {formatCount(summary?.sample_size)} 件</p>
        </div>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-[28px] bg-white/12 border border-white/25 p-6 shadow-inner">
          <p className="text-xs uppercase tracking-[0.3em] text-white/70 mb-2">最近抜けたタグ</p>
          <p className="text-4xl font-black leading-tight">
            {highlightTag ? `#${highlightTag.tag_name}` : 'データ取得中'}
          </p>
          <div className="mt-4 text-sm text-white/70 space-y-1">
            <p>気になるシェア {formatPercent(highlightTag?.share)}</p>
            <p>最新の気になる {formatDate(highlightTag?.last_liked_at)}</p>
            <p>代表作 {highlightTag?.representative_video?.title ?? '—'}</p>
          </div>
        </div>
        <div className="rounded-[28px] bg-gradient-to-br from-rose-500/70 via-fuchsia-500/60 to-indigo-500/60 border border-white/20 p-6 shadow-lg flex flex-col justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-white/80 mb-2">抜ける出演者</p>
            <p className="text-4xl font-black leading-tight">
              {highlightPerformer ? highlightPerformer.performer_name : 'データ取得中'}
            </p>
            <div className="mt-4 text-sm text-white/85 space-y-1">
              <p>気になるシェア {formatPercent(highlightPerformer?.share)}</p>
              <p>最新の気になる {formatDate(highlightPerformer?.last_liked_at)}</p>
            </div>
          </div>
          <p className="text-sm text-white/90 mt-4">{highlightCopy}</p>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
        <div className="rounded-2xl bg-white/8 border border-white/15 p-5 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-white/85">Top Tags</p>
            <span className="text-[10px] uppercase tracking-[0.4em] text-white/50">trend</span>
          </div>
          {tags.length === 0 ? (
            <p className="text-xs text-white/60">「気になる」が集まるとランキングが表示されます。</p>
          ) : (
            <div className="flex flex-col gap-3">
              {tags.map((tag, index) => (
                <div key={tag.tag_id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-bold text-white/60 w-6 text-right">{`#${index + 1}`}</span>
                    <div>
                      <p className="font-semibold text-base">{tag.tag_name}</p>
                      <p className="text-[11px] text-white/60">
                        気になる {formatCount(tag.likes)} / スキップ {formatCount(tag.nopes)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right text-xs text-white/70">
                    <p className="uppercase tracking-wide text-[10px]">share</p>
                    <p className="text-base font-bold">{formatPercent(tag.share)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-2xl bg-white/8 border border-white/15 p-5 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-white/85">Top Performers</p>
            <span className="text-[10px] uppercase tracking-[0.4em] text-white/50">bias</span>
          </div>
          {performers.length === 0 ? (
            <p className="text-xs text-white/60">「気になる」とした作品の出演者がここに並びます。</p>
          ) : (
            <div className="flex flex-col gap-3">
              {performers.map((performer, index) => (
                <div key={performer.performer_id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-bold text-white/60 w-6 text-right">{`#${index + 1}`}</span>
                    <div>
                      <p className="font-semibold text-base">{performer.performer_name}</p>
                      <p className="text-[11px] text-white/60">
                        気になる {formatCount(performer.likes)} / スキップ {formatCount(performer.nopes)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right text-xs text-white/70">
                    <p className="uppercase tracking-wide text-[10px]">share</p>
                    <p className="text-base font-bold">{formatPercent(performer.share)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <footer className="flex items-center justify-between text-xs text-white/70">
        <p>「気になる」履歴 ({formatCount(summary?.sample_size)} 件) をもとに作成</p>
        <p>#あなたの性癖 を診断してシェア</p>
      </footer>
    </div>
  );
});

export default AnalysisShareCard;

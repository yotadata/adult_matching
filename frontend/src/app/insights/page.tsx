'use client';

import { useMemo, useRef, useState } from 'react';
import { Share2, Info, Smile, Frown, TrendingUp, Tag as TagIcon, Users, Clock } from 'lucide-react';
import { toPng } from 'html-to-image';
import ShareModal from '@/components/ShareModal';
import AnalysisShareCard from '@/components/AnalysisShareCard';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';

const WINDOW_OPTIONS: Array<{ label: string; value: number | null }> = [
  { label: '3日', value: 3 },
  { label: '7日', value: 7 },
  { label: '30日', value: 30 },
  { label: '全期間', value: null },
];

const MAX_FETCH = 2000;
const SHARE_CARD_LANDING_URL = 'https://seihekilab.com/?utm_source=share_card&utm_medium=user_profile';

const formatPercent = (value: number | null): string => {
  if (value === null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
};

const formatDate = (value: string | null): string => {
  if (!value) return '—';
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return new Intl.DateTimeFormat('ja-JP', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(date);
  } catch {
    return '—';
  }
};

const formatCount = (value: number): string => value.toLocaleString('ja-JP');

function SummaryCard({
  title,
  value,
  sub,
  icon: Icon,
}: {
  title: string;
  value: string;
  sub?: string;
  icon: React.ComponentType<{ size?: number }>;
}) {
  return (
    <div className="rounded-xl bg-white/80 backdrop-blur shadow-lg border border-white/40 p-4 text-gray-900 flex items-center gap-4">
      <div className="flex items-center justify-center w-12 h-12 rounded-full bg-rose-100 text-rose-500 shadow-inner">
        <Icon size={22} />
      </div>
      <div className="flex flex-col">
        <span className="text-sm font-semibold text-gray-500">{title}</span>
        <span className="text-2xl font-bold">{value}</span>
        {sub ? <span className="text-xs text-gray-500 mt-1">{sub}</span> : null}
      </div>
    </div>
  );
}

export default function AnalysisResultsPage() {
  const [windowDays, setWindowDays] = useState<number | null>(null);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [shareImageUrl, setShareImageUrl] = useState<string | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const { data, loading, error } = useAnalysisResults({
    windowDays,
    tagLimit: 6,
    performerLimit: 6,
    recentLimit: 12,
  });

  const summary = data?.summary ?? null;
  const topTags = data?.top_tags ?? [];
  const topPerformers = data?.top_performers ?? [];
  const recent = data?.recent_decisions ?? [];

  const windowLabel = useMemo(() => {
    if (!summary) return '';
    if (summary.window_days === null) return '全期間';
    return `直近 ${summary.window_days} 日`;
  }, [summary]);

  const shareUrl = typeof window !== 'undefined' ? window.location.href : 'https://seiheki.me/insights';
  const primaryTag = topTags[0];
  const secondaryTag = topTags[1];
  const primaryPerformer = topPerformers[0];
  const shareTagNames = [primaryTag, secondaryTag].flatMap((tag) => (tag ? [`#${tag.tag_name}`] : []));
  const sharePerformerNames = primaryPerformer ? [primaryPerformer.performer_name] : [];
  const shareText = (() => {
    if (primaryTag && primaryPerformer) {
      return `最近は ${primaryPerformer.performer_name} が出る #${primaryTag.tag_name} 系で毎回「気になる」。あなたも #あなたの性癖 を診断して抜けるポイントをシェアしよう。`;
    }
    if (shareTagNames.length > 0) {
      return `今の抜けるタグは ${shareTagNames.join(' / ')}。あなたも #あなたの性癖 を診断して好みカードを作ろう。`;
    }
    if (sharePerformerNames.length > 0) {
      return `${sharePerformerNames.join(' / ')} が出ていたら即「気になる」。あなたも #あなたの性癖 を診断して抜ける出演者を布教しよう。`;
    }
    if (summary) {
      return `${windowLabel}の好み分析結果をシェアしました。あなたも #あなたの性癖 を診断してみて。`;
    }
    return 'あなたの性癖の結果をシェアしました。';
  })();

  const handleShare = async () => {
    if (!cardRef.current || !summary) {
      if (typeof window !== 'undefined') {
        window.alert('データを取得してからシェアしてください。');
      }
      return;
    }
    try {
      const dataUrl = await toPng(cardRef.current, {
        cacheBust: true,
        pixelRatio: 2,
        backgroundColor: '#050113',
      });
      setShareImageUrl(dataUrl);
      setIsShareModalOpen(true);
    } catch (err) {
      console.error('Failed to generate share image', err);
      if (typeof navigator !== 'undefined' && navigator.share) {
        try {
          await navigator.share({
            title: 'あなたの性癖分析結果',
            text: shareText,
            url: shareUrl,
          });
        } catch {
          /* noop */
        }
      }
    }
  };

  const handleOpenProduct = (url?: string | null) => {
    if (!url) return;
    try {
      window.open(url, '_blank', 'noopener,noreferrer');
    } catch {
      /* noop */
    }
  };

  return (
    <>
      <main className="w-full min-h-screen px-0 sm:px-4 py-8">
      <section className="w-full mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 text-white">
        <header className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight">あなたの性癖</h1>
            <p className="text-sm text-white/80">「気になる」 / 「スキップ」の履歴から好みの傾向を可視化します。</p>
          </div>
          <button
            onClick={handleShare}
            aria-label="共有"
            className="inline-flex items-center gap-1 rounded-md bg-white/20 hover:bg-white/30 backdrop-blur px-3 py-2 text-sm font-semibold text-white transition"
            title="共有"
          >
            <Share2 size={16} />
            シェア（β版）
          </button>
        </header>

        <div className="flex flex-col lg:flex-row gap-4 lg:items-center lg:justify-between mb-6">
          <div className="flex flex-wrap gap-2">
            {WINDOW_OPTIONS.map((option) => {
              const active = option.value === windowDays || (option.value === null && windowDays === null);
              return (
                <button
                  key={option.label}
                  onClick={() => setWindowDays(option.value)}
                  className={`px-3 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition ${
                    active ? 'bg-white text-rose-500 shadow-lg' : 'bg-white/20 text-white hover:bg-white/30'
                  }`}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
        </div>

        {loading ? (
          <div className="grid gap-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Array.from({ length: 3 }).map((_, idx) => (
                <div key={`summary-skeleton-${idx}`} className="h-24 rounded-xl bg-white/30 animate-pulse" />
              ))}
            </div>
            <div className="h-32 rounded-xl bg-white/30 animate-pulse" />
          </div>
        ) : (
          <>
            {error ? (
              <div className="rounded-xl bg-red-500/20 border border-red-400/60 text-white px-4 py-3 mb-6">
                {error}
              </div>
            ) : null}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <SummaryCard
                title="気になる数"
                value={summary ? formatCount(summary.total_likes) : '—'}
                sub={summary ? `${windowLabel}の「気になる」合計` : undefined}
                icon={Smile}
              />
              <SummaryCard
                title="スキップ数"
                value={summary ? formatCount(summary.total_nope) : '—'}
                sub={summary ? `${windowLabel}の「スキップ」合計` : undefined}
                icon={Frown}
              />
              <SummaryCard
                title="気になる比率"
                value={summary ? formatPercent(summary.like_ratio) : '—'}
                sub={summary ? `サンプル ${formatCount(summary.sample_size)} 件` : undefined}
                icon={TrendingUp}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
              <section className="rounded-xl bg-white/85 backdrop-blur-md shadow-lg border border-white/50 p-5 text-gray-900">
                <header className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <h2 className="text-lg font-bold flex items-center gap-2">
                      <TagIcon size={18} className="text-rose-400" />
                      好きなタグ Top {topTags.length}
                    </h2>
                    <p className="text-xs text-gray-500 mt-1">「気になる」が多いタグの傾向を表示します。</p>
                  </div>
                  <Info size={16} className="text-gray-400 shrink-0" />
                </header>
                {topTags.length === 0 ? (
                  <div className="text-sm text-gray-500 bg-white/60 rounded-lg p-4">
                    データがまだありません。作品に「気になる」を付けるとタグ分析が表示されます。
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">
                    {topTags.map((tag) => (
                      <div key={tag.tag_id} className="rounded-lg border border-gray-200 bg-white/80 shadow-sm p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <div className="flex flex-col">
                          <span className="text-sm font-semibold text-gray-700">{tag.tag_name}</span>
                          <span className="text-xs text-gray-500">
                            気になる: {formatCount(tag.likes)} / スキップ: {formatCount(tag.nopes)}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <div className="flex flex-col text-right">
                            <span>気になる比率</span>
                            <span className="text-base font-bold text-gray-900">{formatPercent(tag.like_ratio)}</span>
                          </div>
                          <div className="flex flex-col text-right">
                            <span>気になるシェア</span>
                            <span className="text-base font-bold text-gray-900">{tag.share != null ? formatPercent(tag.share) : '—'}</span>
                          </div>
                          {tag.representative_video ? (
                            <button
                              onClick={() => handleOpenProduct(tag.representative_video?.product_url)}
                              className="px-3 py-1.5 text-xs font-semibold rounded-full bg-rose-500/15 text-rose-500 hover:bg-rose-500/25 transition"
                            >
                              代表作を見る
                            </button>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="rounded-xl bg-white/85 backdrop-blur-md shadow-lg border border-white/50 p-5 text-gray-900">
                <header className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <h2 className="text-lg font-bold flex items-center gap-2">
                      <Users size={18} className="text-rose-400" />
                      好きな出演者 Top {topPerformers.length}
                    </h2>
                    <p className="text-xs text-gray-500 mt-1">「気になる」とした作品に多く登場する出演者の傾向を表示します。</p>
                  </div>
                  <Info size={16} className="text-gray-400 shrink-0" />
                </header>
                {topPerformers.length === 0 ? (
                  <div className="text-sm text-gray-500 bg-white/60 rounded-lg p-4">
                    データがまだありません。「気になる」とした作品の出演者がここに表示されます。
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">
                    {topPerformers.map((performer) => (
                      <div key={performer.performer_id} className="rounded-lg border border-gray-200 bg-white/80 shadow-sm p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <div className="flex flex-col">
                          <span className="text-sm font-semibold text-gray-700">{performer.performer_name}</span>
                          <span className="text-xs text-gray-500">
                            気になる: {formatCount(performer.likes)} / スキップ: {formatCount(performer.nopes)}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <div className="flex flex-col text-right">
                            <span>気になる比率</span>
                            <span className="text-base font-bold text-gray-900">{formatPercent(performer.like_ratio)}</span>
                          </div>
                          <div className="flex flex-col text-right">
                            <span>気になるシェア</span>
                            <span className="text-base font-bold text-gray-900">{performer.share != null ? formatPercent(performer.share) : '—'}</span>
                          </div>
                          {performer.representative_video ? (
                            <button
                              onClick={() => handleOpenProduct(performer.representative_video?.product_url)}
                              className="px-3 py-1.5 text-xs font-semibold rounded-full bg-rose-500/15 text-rose-500 hover:bg-rose-500/25 transition"
                            >
                              出演作を見る
                            </button>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>

            <section className="rounded-xl bg-white/85 backdrop-blur-md shadow-lg border border-white/50 p-5 text-gray-900 mb-6">
              <header className="flex items-start justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-bold flex items-center gap-2">
                    <Clock size={18} className="text-rose-400" />
                    直近の判断履歴
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">最新の「気になる」 / 「スキップ」を新しい順に表示します。</p>
                </div>
                <Info size={16} className="text-gray-400 shrink-0" />
              </header>
              {recent.length === 0 ? (
                <div className="text-sm text-gray-500 bg-white/60 rounded-lg p-4">
                  まだ判断履歴がありません。スワイプ画面で評価するとここに表示されます。
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full table-fixed border-separate border-spacing-y-2 text-sm text-gray-700">
                    <thead>
                      <tr className="text-xs uppercase tracking-wide text-gray-500">
                        <th className="px-3 py-2 text-left font-semibold w-[32%]">作品</th>
                        <th className="px-3 py-2 text-left font-semibold w-[12%]">判断</th>
                        <th className="px-3 py-2 text-left font-semibold w-[20%]">タグ</th>
                        <th className="px-3 py-2 text-left font-semibold w-[20%]">出演者</th>
                        <th className="px-3 py-2 text-left font-semibold w-[12%]">決定日</th>
                        <th className="px-3 py-2 text-left font-semibold w-[8%]">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recent.map((item) => (
                        <tr
                          key={`${item.video_id}-${item.decided_at}`}
                          className="bg-white/80 border border-gray-200/80 rounded-xl shadow-sm align-top"
                        >
                          <td className="px-3 py-3 align-top">
                            <div className="flex items-start gap-3">
                              <div className="h-14 w-20 shrink-0 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
                                {item.thumbnail_url ? (
                                  // eslint-disable-next-line @next/next/no-img-element
                                  <img src={item.thumbnail_url} alt={item.title ?? 'thumbnail'} className="w-full h-full object-cover" />
                                ) : null}
                              </div>
                              <div className="flex flex-col gap-1 w-full min-w-0">
                                <span className="text-sm font-semibold text-gray-900 line-clamp-3">{item.title ?? 'タイトル未設定'}</span>
                              </div>
                            </div>
                          </td>
                          <td className="px-3 py-3 align-top">
                            <span
                              className={`inline-flex items-center justify-center min-w-[64px] px-2 py-0.5 text-xs font-bold rounded-full ${
                                item.decision_type === 'like' ? 'bg-rose-500/15 text-rose-500' : 'bg-gray-200 text-gray-700'
                              }`}
                            >
                              {item.decision_type === 'like' ? '気になる' : 'スキップ'}
                            </span>
                          </td>
                          <td className="px-3 py-3 align-top">
                            <div className="flex flex-wrap gap-1">
                              {item.tags.slice(0, 4).map((tag) => (
                                <span key={tag.id} className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200/60 text-xs">
                                  #{tag.name}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-3 align-top">
                            <div className="flex flex-wrap gap-1">
                              {item.performers.slice(0, 4).map((perf) => (
                                <span key={perf.id} className="px-1.5 py-0.5 rounded-full bg-pink-100 text-pink-600 border border-pink-200/60 text-xs">
                                  {perf.name}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-3 align-top text-xs text-gray-600">
                            {formatDate(item.decided_at)}
                          </td>
                          <td className="px-3 py-3 align-top min-w-[110px]">
                            {item.product_url ? (
                              <button
                                onClick={() => handleOpenProduct(item.product_url)}
                                className="px-3 py-1.5 text-xs font-semibold rounded-full bg-rose-500/15 text-rose-500 hover:bg-rose-500/25 transition"
                              >
                                作品ページへ
                              </button>
                            ) : (
                              <span className="text-xs text-gray-400">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <section className="rounded-xl bg-white/85 backdrop-blur-md shadow-lg border border-white/50 p-5 text-gray-900">
              <header className="flex items-start gap-3 mb-4">
                <Info size={18} className="text-rose-400 mt-0.5" />
                <div>
                  <h2 className="text-lg font-bold">集計メモ</h2>
                  <p className="text-xs text-gray-500 mt-1">
                    集計対象は {windowLabel} に行った「気になる」 / 「スキップ」で、最大 {MAX_FETCH.toLocaleString('ja-JP')} 件までを対象にしています。
                    タグ・出演者のシェアは「気になる」件数を母数に算出されます。
                  </p>
                </div>
              </header>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-gray-600">
                <div className="rounded-lg border border-gray-200 bg-white/80 p-4">
                  <div className="text-xs font-semibold text-gray-500 mb-1">最初の判断</div>
                  <div className="text-base font-bold text-gray-900">{summary ? formatDate(summary.first_decision_at) : '—'}</div>
                </div>
                <div className="rounded-lg border border-gray-200 bg-white/80 p-4">
                  <div className="text-xs font-semibold text-gray-500 mb-1">最新の判断</div>
                  <div className="text-base font-bold text-gray-900">{summary ? formatDate(summary.latest_decision_at) : '—'}</div>
                </div>
              </div>
            </section>
          </>
        )}
      </section>
      </main>
      <ShareModal
        isOpen={isShareModalOpen}
        onClose={() => setIsShareModalOpen(false)}
        imageUrl={shareImageUrl}
        shareUrl={shareUrl}
        shareText={shareText}
      />
      <div className="absolute -left-[9999px] top-0 pointer-events-none select-none" aria-hidden>
        <AnalysisShareCard
          ref={cardRef}
          summary={summary ?? null}
          topTags={topTags}
          topPerformers={topPerformers}
          shareUrl={SHARE_CARD_LANDING_URL}
        />
      </div>
    </>
  );
}

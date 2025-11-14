'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Sparkles, RefreshCcw, Info, Play, Loader2, Clapperboard, Search } from 'lucide-react';
import { useAiRecommend, type AiRecommendSection } from '@/hooks/useAiRecommend';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';

const QUICK_PROMPTS = ['甘め', '刺激的', '癒やし', '探索', 'パートナーと'];
const SEARCH_SHORTCUTS = [
  {
    title: 'スワイプでサクッと探す',
    description: 'テンポ良く LIKE/NOPE しながら直感的に選びたいとき',
    href: '/swipe',
    badge: 'スワイプ',
  },
  {
    title: 'タグ・出演者で深掘り',
    description: '嗜好分析ページで最近の好みや人気タグを確認しながら探す',
    href: '/analysis-results',
    badge: '嗜好分析',
  },
  {
    title: 'LIKE済みを整理',
    description: 'これまでのいいね作品を一覧で眺めて気分に合うものを再生',
    href: '/likes',
    badge: 'LIKE一覧',
  },
];

const formatDuration = (minutes: number | null) => {
  if (!minutes) return '—';
  if (minutes < 60) return `${minutes}分`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest === 0 ? `${hours}時間` : `${hours}時間${rest}分`;
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return '—';
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return date.toLocaleDateString('ja-JP');
  } catch {
    return '—';
  }
};

export default function AiRecommendPage() {
  const [promptInput, setPromptInput] = useState('');
  const [appliedPrompt, setAppliedPrompt] = useState('');
  const [expandedItemId, setExpandedItemId] = useState<string | null>(null);

  const { data, loading, error, refetch } = useAiRecommend({
    prompt: appliedPrompt,
    limitPerSection: 12,
  });

  const { data: analysisData } = useAnalysisResults({
    windowDays: 90,
    includeNope: false,
    tagLimit: 3,
    performerLimit: 3,
    recentLimit: 0,
  });

  const topTags = analysisData?.top_tags ?? [];
  const topPerformers = analysisData?.top_performers ?? [];

  const activePromptText = appliedPrompt ? appliedPrompt : 'キーワード未設定';

  const handleApplyPrompt = () => {
    setAppliedPrompt(promptInput.trim());
    setExpandedItemId(null);
  };

  const handleClearPrompt = () => {
    setPromptInput('');
    setAppliedPrompt('');
    setExpandedItemId(null);
  };

  const handlePromptChip = (keyword: string) => {
    setPromptInput((prev) => {
      if (prev.includes(keyword)) return prev;
      return prev.trim().length > 0 ? `${prev} ${keyword}` : keyword;
    });
  };

  const renderSection = (section: AiRecommendSection) => (
    <section key={section.id} className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-6 flex flex-col gap-4">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <LayersIcon sectionId={section.id} />
            {section.title}
          </h3>
          <p className="text-sm text-gray-600 mt-1">{section.rationale}</p>
        </div>
      </header>
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
        {section.items.map((item) => {
          const composedId = `${section.id}:${item.id}`;
          const isExpanded = expandedItemId === composedId;
          return (
            <article
              key={item.id}
              className="w-full rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col overflow-hidden text-sm"
            >
              <div className="relative w-full aspect-[16/9] bg-gray-200">
                {item.thumbnail_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={item.thumbnail_url} alt={item.title ?? 'thumbnail'} className="absolute inset-0 w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">No Image</div>
                )}
                <div className="absolute top-2 right-2 bg-white/90 text-[11px] px-1.5 py-0.5 rounded-full text-gray-700 capitalize">
                  {item.metrics.source}
                </div>
              </div>
              <div className="flex flex-col gap-2.5 p-3 flex-1">
                <div className="space-y-1">
                  <h4 className="text-[13px] font-semibold text-gray-900 line-clamp-2">{item.title ?? 'タイトル未設定'}</h4>
                  <div className="text-[11px] text-gray-500 flex flex-wrap gap-1">
                    {item.tags.slice(0, 3).map((tag) => (
                      <span key={tag.id} className="px-1 py-0.5 bg-gray-100 rounded-full border border-gray-200 text-gray-600">
                        #{tag.name}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="space-y-1 text-[12px] text-gray-500">
                  <p className="leading-tight">{item.reason.summary}</p>
                  <button
                    type="button"
                    onClick={() => {
                      setExpandedItemId(isExpanded ? null : composedId);
                    }}
                    className="inline-flex items-center gap-1 text-rose-500 hover:text-rose-400 font-semibold text-[11px]"
                  >
                    <Info size={14} />
                    {isExpanded ? '閉じる' : '詳細を見る'}
                  </button>
                  {isExpanded ? (
                    <div className="mt-1 rounded-md bg-gray-50 border border-gray-200 p-2 text-[11px] text-gray-600 space-y-1">
                      <p>{item.reason.detail}</p>
                      {item.reason.highlights.length ? (
                        <p className="text-gray-500">ハイライト: {item.reason.highlights.join(' ')}</p>
                      ) : null}
                      {item.metrics.score ? (
                        <p>適合度 {Math.round((item.metrics.score ?? 0) * 100)}%</p>
                      ) : null}
                      {item.metrics.popularity_score ? (
                        <p>人気指標 {item.metrics.popularity_score.toLocaleString('ja-JP')}</p>
                      ) : null}
                      {item.metrics.product_released_at ? (
                        <p>リリース日 {formatDate(item.metrics.product_released_at)}</p>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                <div className="mt-auto flex items-center justify-between text-[11px] text-gray-500">
                  <span>{formatDuration(item.duration_minutes ?? null)}</span>
                  <button
                    type="button"
                    onClick={() => {
                      if (item.product_url) window.open(item.product_url, '_blank', 'noopener,noreferrer');
                    }}
                    className="inline-flex items-center gap-1 text-gray-600 hover:text-gray-800"
                  >
                    <Play size={14} />
                    作品ページへ
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );

  const isEmpty = !loading && !error && (data?.sections.length ?? 0) === 0;

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <section className="w-full rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 flex flex-col gap-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-white">
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight">AIで探す</h1>
              <p className="text-sm text-white/80 mt-2">
                あなたのLIKE履歴・全体トレンド・今の気分キーワードをもとに、3種類のリストを自動生成します。
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setExpandedItemId(null);
                refetch();
              }}
              className="inline-flex items-center gap-2 rounded-md bg-white/20 hover:bg-white/30 px-3 py-2 text-sm font-semibold transition"
            >
              <RefreshCcw size={16} />
              再取得
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-4">
            <div className="space-y-4">
              <div className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
                <header>
                  <p className="text-xs uppercase tracking-[0.4em] text-rose-500/70">今の気分</p>
                  <h2 className="text-xl font-bold text-gray-900 mt-1">キーワードを伝えて調整</h2>
                  <p className="text-sm text-gray-600 mt-1">気分やシチュエーションを入力すると、3つ目のセクションで優先的に反映されます。</p>
                </header>
                <div className="space-y-3">
                  <textarea
                    value={promptInput}
                    onChange={(e) => setPromptInput(e.target.value)}
                    placeholder="例: 甘めで癒やされたい / 今日は刺激多め / パートナーと観られるもの"
                    rows={3}
                    className="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-rose-400"
                  />
                  <div className="flex flex-wrap gap-2">
                    {QUICK_PROMPTS.map((keyword) => (
                      <button
                        key={keyword}
                        type="button"
                        onClick={() => handlePromptChip(keyword)}
                        className="px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-500/15 text-rose-500 hover:bg-rose-500/25"
                      >
                        {keyword}
                      </button>
                    ))}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleApplyPrompt}
                      className="inline-flex items-center gap-2 rounded-md bg-rose-500 text-white px-4 py-2 text-sm font-semibold hover:bg-rose-600"
                    >
                      <Sparkles size={16} />
                      キーワードを適用
                    </button>
                    <button
                      type="button"
                      onClick={handleClearPrompt}
                      className="inline-flex items-center gap-2 rounded-md border border-white/40 bg-white/10 text-white px-4 py-2 text-sm font-semibold hover:bg-white/20"
                    >
                      クリア
                    </button>
                  </div>
                  <p className="text-xs text-gray-600">現在の入力: {activePromptText}</p>
                </div>
              </div>
            </div>

            <aside className="rounded-2xl bg-white/80 border border-white/60 shadow-lg p-5 flex flex-col gap-4 text-gray-900">
              <header>
                <h3 className="text-base font-bold flex items-center gap-2">
                  <Sparkles size={16} className="text-rose-500" />
                  最近の嗜好スナップショット
                </h3>
                <p className="text-xs text-gray-500 mt-1">直近90日のLIKE履歴から抽出</p>
              </header>
              <div className="space-y-3 text-sm text-gray-700">
                <div>
                  <p className="text-xs text-gray-500">トップタグ</p>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {topTags.length === 0 ? (
                      <span className="text-xs text-gray-400">データなし</span>
                    ) : (
                      topTags.map((tag) => (
                        <span key={tag.tag_id} className="px-2 py-1 rounded-full bg-rose-500/10 text-rose-500 border border-rose-200 text-xs">
                          #{tag.tag_name}
                        </span>
                      ))
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-500">よく観ている出演者</p>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {topPerformers.length === 0 ? (
                      <span className="text-xs text-gray-400">データなし</span>
                    ) : (
                      topPerformers.map((performer) => (
                        <span key={performer.performer_id} className="px-2 py-1 rounded-full bg-indigo-500/10 text-indigo-500 border border-indigo-200 text-xs">
                          {performer.performer_name}
                        </span>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </section>

        {error ? (
          <div className="rounded-2xl bg-red-500/10 border border-red-400/40 text-red-600 px-4 py-3">
            {error}
          </div>
        ) : null}

        {loading ? (
          <div className="grid gap-4">
            {Array.from({ length: 3 }).map((_, idx) => (
              <div key={`skeleton-${idx}`} className="h-64 rounded-2xl bg-white/10 backdrop-blur border border-white/20 animate-pulse" />
            ))}
          </div>
        ) : isEmpty ? (
          <div className="rounded-2xl bg-white/80 border border-white/60 shadow-lg p-6 text-center text-gray-600">
            候補を取得できませんでした。キーワードを変えて再度お試しください。
          </div>
        ) : (
          data?.sections.map(renderSection)
        )}

        {!loading && (
          <section className="rounded-2xl bg-white/85 border border-white/60 shadow-lg p-6 flex flex-col gap-4">
            <header className="flex flex-col gap-1">
              <p className="text-xs uppercase tracking-[0.35em] text-gray-400">検索ショートカット</p>
              <h3 className="text-lg font-bold text-gray-900">もっと探したいときは</h3>
              <p className="text-sm text-gray-600">AI棚で気になったあとに、従来の探し方へすぐ移れます。</p>
            </header>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {SEARCH_SHORTCUTS.map((shortcut) => (
                <Link
                  key={shortcut.href}
                  href={shortcut.href}
                  className="group rounded-2xl border border-gray-200 bg-white/80 p-4 flex flex-col gap-2 hover:border-gray-400 transition"
                >
                  <div className="flex items-center gap-2 text-xs font-semibold text-rose-500">
                    <Search size={14} />
                    {shortcut.badge}
                  </div>
                  <h4 className="text-base font-bold text-gray-900 group-hover:text-gray-700">{shortcut.title}</h4>
                  <p className="text-sm text-gray-600">{shortcut.description}</p>
                </Link>
              ))}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

function LayersIcon({ sectionId }: { sectionId: string }) {
  if (sectionId.includes('prompt')) return <Sparkles size={18} className="text-rose-500" />;
  if (sectionId.includes('trend')) return <RefreshCcw size={18} className="text-indigo-500" />;
  if (sectionId.includes('fresh')) return <Clapperboard size={18} className="text-orange-500" />;
  if (sectionId.includes('fallback')) return <Loader2 size={18} className="text-gray-500" />;
  return <Loader2 size={18} className="text-emerald-500" />;
}

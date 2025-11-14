'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Sparkles, RefreshCcw, Info, Play, Loader2, Clapperboard, Search } from 'lucide-react';
import { Combobox } from '@headlessui/react';
import { useAiRecommend, type AiRecommendSection } from '@/hooks/useAiRecommend';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';
import { supabase } from '@/lib/supabase';

const SEARCH_SHORTCUTS = [
  {
    title: 'スワイプでサクッと探す',
    description: 'テンポ良く LIKE/NOPE しながら直感的に選びたいとき',
    href: '/swipe',
    badge: 'スワイプ',
  },
  {
    title: '嗜好レポートを見る',
    description: '最近の傾向やランキングを可視化しながら条件を考える',
    href: '/insights',
    badge: 'インサイト',
  },
  {
    title: '気になるリストをまとめる',
    description: 'AIやスワイプで追加した「気になる」を一括で振り返る',
    href: '/lists',
    badge: 'リスト管理',
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
  const [expandedItemId, setExpandedItemId] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedPerformers, setSelectedPerformers] = useState<Array<{ id: string; name: string }>>([]);
  const [appliedTagIds, setAppliedTagIds] = useState<string[]>([]);
  const [appliedPerformerIds, setAppliedPerformerIds] = useState<string[]>([]);
  const [tagSearchTerm, setTagSearchTerm] = useState('');
  const [performerSearchTerm, setPerformerSearchTerm] = useState('');
  const [tagSearchResults, setTagSearchResults] = useState<Array<{ id: string; name: string }>>([]);
  const [performerSearchResults, setPerformerSearchResults] = useState<Array<{ id: string; name: string }>>([]);
  const [tagLookupLoading, setTagLookupLoading] = useState(false);
  const [performerLookupLoading, setPerformerLookupLoading] = useState(false);

  const { data, loading, error } = useAiRecommend({
    limitPerSection: 12,
    tagIds: appliedTagIds,
    performerIds: appliedPerformerIds,
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
  useEffect(() => {
    const term = tagSearchTerm.trim();
    if (term.length < 2) {
      setTagSearchResults([]);
      setTagLookupLoading(false);
      return;
    }
    let cancelled = false;
    setTagLookupLoading(true);
    const timer = setTimeout(async () => {
      try {
        const { data } = await supabase
          .from('tags')
          .select('id,name')
          .ilike('name', `%${term}%`)
          .order('name', { ascending: true })
          .limit(8);
        if (cancelled) return;
        setTagSearchResults((data ?? []).map((item) => ({ id: String(item.id), name: (item as { name?: string }).name ?? '' })));
      } catch {
        if (cancelled) return;
        setTagSearchResults([]);
      } finally {
        if (!cancelled) setTagLookupLoading(false);
      }
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tagSearchTerm]);

  useEffect(() => {
    const term = performerSearchTerm.trim();
    if (term.length < 2) {
      setPerformerSearchResults([]);
      setPerformerLookupLoading(false);
      return;
    }
    let cancelled = false;
    setPerformerLookupLoading(true);
    const timer = setTimeout(async () => {
      try {
        const { data } = await supabase
          .from('performers')
          .select('id,name')
          .ilike('name', `%${term}%`)
          .order('name', { ascending: true })
          .limit(8);
        if (cancelled) return;
        setPerformerSearchResults(
          (data ?? []).map((item) => ({ id: String(item.id), name: (item as { name?: string }).name ?? '' })),
        );
      } catch {
        if (cancelled) return;
        setPerformerSearchResults([]);
      } finally {
        if (!cancelled) setPerformerLookupLoading(false);
      }
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [performerSearchTerm]);

  const handleApplyFilters = () => {
    setAppliedTagIds(selectedTags.map((tag) => tag.id));
    setAppliedPerformerIds(selectedPerformers.map((perf) => perf.id));
    setExpandedItemId(null);
  };

  const handleClearFilters = () => {
    setSelectedTags([]);
    setSelectedPerformers([]);
    setAppliedTagIds([]);
    setAppliedPerformerIds([]);
    setTagSearchTerm('');
    setPerformerSearchTerm('');
    setTagSearchResults([]);
    setPerformerSearchResults([]);
    setExpandedItemId(null);
  };

  const toggleTagSelection = (option: { id: string; name: string }) => {
    setSelectedTags((prev) => {
      if (prev.some((tag) => tag.id === option.id)) return prev.filter((tag) => tag.id !== option.id);
      if (prev.length >= 5) return prev;
      return [...prev, option];
    });
  };

  const togglePerformerSelection = (option: { id: string; name: string }) => {
    setSelectedPerformers((prev) => {
      if (prev.some((perf) => perf.id === option.id)) return prev.filter((perf) => perf.id !== option.id);
      if (prev.length >= 5) return prev;
      return [...prev, option];
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
      <div className="flex gap-3 overflow-x-auto pb-2 snap-x snap-mandatory">
        {section.items.map((item) => {
          const composedId = `${section.id}:${item.id}`;
          const isExpanded = expandedItemId === composedId;
          return (
            <article
              key={item.id}
              className="min-w-[220px] max-w-[220px] rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col overflow-hidden text-sm snap-start"
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
          <div className="flex flex-col gap-3 text-white">
            <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight">AIで探す</h1>
            <p className="text-sm text-white/80">
              あなたのLIKE履歴・全体トレンド・今の気分キーワードをもとに、3種類のリストを自動生成します。
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-4">
            <div className="space-y-4">
              <div className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
                <header className="space-y-1">
                  <p className="text-xs uppercase tracking-[0.4em] text-rose-500/70">今の気分</p>
                  <h2 className="text-xl font-bold text-gray-900 mt-1">キーワードを伝えて調整</h2>
                  <p className="text-sm text-gray-600">
                    下の検索欄からタグと出演者をそれぞれ最大5件まで選択すると、横並びのリストがその条件で入れ替わります。プリセットから選ぶか、キーワード検索で全体から探してください。
                  </p>
                </header>
                <div className="space-y-4">
                  <div className="space-y-3">
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 font-semibold">タグを検索（全体から）</p>
                      <Combobox<{ id: string; name: string } | null>
                        value={null}
                        onChange={(value) => {
                          if (value) toggleTagSelection(value);
                        }}
                      >
                        <div className="relative">
                          <div className="flex items-center gap-2 rounded-2xl border border-gray-200 bg-white px-3 py-2 shadow-sm">
                            <Search size={16} className="text-rose-400" />
                            <Combobox.Input
                              className="w-full text-sm text-gray-900 placeholder-gray-400 focus:outline-none"
                              placeholder="タグ名を入力（例: 制服, OL など）"
                              onChange={(event) => setTagSearchTerm(event.target.value)}
                              displayValue={() => ''}
                            />
                          </div>
                          <Combobox.Options className="absolute z-20 mt-2 max-h-48 w-full overflow-auto rounded-2xl border border-gray-200 bg-white text-sm shadow-lg">
                            {tagSearchTerm.trim().length < 2 ? (
                              <div className="px-3 py-2 text-xs text-gray-400">2文字以上入力すると候補が表示されます。</div>
                            ) : tagLookupLoading ? (
                              <div className="px-3 py-2 text-xs text-gray-400">タグを検索中…</div>
                            ) : tagSearchResults.length === 0 ? (
                              <div className="px-3 py-2 text-xs text-gray-400">候補が見つかりません</div>
                            ) : (
                              tagSearchResults.map((tag) => (
                                <Combobox.Option
                                  key={`combobox-tag-${tag.id}`}
                                  value={tag}
                                  className={({ active }) =>
                                    `flex items-center justify-between px-3 py-1.5 text-sm ${active ? 'bg-rose-50 text-rose-600' : 'text-gray-700'}`
                                  }
                                >
                                  <span>#{tag.name}</span>
                                  {selectedTags.some((t) => t.id === tag.id) ? (
                                    <span className="text-[10px] text-rose-500">選択中</span>
                                  ) : null}
                                </Combobox.Option>
                              )))
                            }
                          </Combobox.Options>
                        </div>
                      </Combobox>
                    </div>

                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 font-semibold">出演者を検索（全体から）</p>
                      <Combobox<{ id: string; name: string } | null>
                        value={null}
                        onChange={(value) => {
                          if (value) togglePerformerSelection(value);
                        }}
                      >
                        <div className="relative">
                          <div className="flex items-center gap-2 rounded-2xl border border-gray-200 bg-white px-3 py-2 shadow-sm">
                            <Search size={16} className="text-indigo-400" />
                            <Combobox.Input
                              className="w-full text-sm text-gray-900 placeholder-gray-400 focus:outline-none"
                              placeholder="出演者名を入力（例: ○○○）"
                              onChange={(event) => setPerformerSearchTerm(event.target.value)}
                              displayValue={() => ''}
                            />
                          </div>
                          <Combobox.Options className="absolute z-20 mt-2 max-h-48 w-full overflow-auto rounded-2xl border border-gray-200 bg-white text-sm shadow-lg">
                            {performerSearchTerm.trim().length < 2 ? (
                              <div className="px-3 py-2 text-xs text-gray-400">2文字以上入力すると候補が表示されます。</div>
                            ) : performerLookupLoading ? (
                              <div className="px-3 py-2 text-xs text-gray-400">出演者を検索中…</div>
                            ) : performerSearchResults.length === 0 ? (
                              <div className="px-3 py-2 text-xs text-gray-400">候補が見つかりません</div>
                            ) : (
                              performerSearchResults.map((performer) => (
                                <Combobox.Option
                                  key={`combobox-perf-${performer.id}`}
                                  value={performer}
                                  className={({ active }) =>
                                    `flex items-center justify-between px-3 py-1.5 text-sm ${active ? 'bg-indigo-50 text-indigo-600' : 'text-gray-700'}`
                                  }
                                >
                                  <span>{performer.name}</span>
                                  {selectedPerformers.some((p) => p.id === performer.id) ? (
                                    <span className="text-[10px] text-indigo-500">選択中</span>
                                  ) : null}
                                </Combobox.Option>
                              )))
                            }
                          </Combobox.Options>
                        </div>
                      </Combobox>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2">
                      {selectedTags.map((tag) => (
                        <span
                          key={`pill-tag-${tag.id}`}
                          className="px-3 py-1 rounded-full text-xs font-semibold border border-rose-400 bg-rose-50 text-rose-600 truncate"
                        >
                          #{tag.name}
                        </span>
                      ))}
                      {selectedPerformers.map((performer) => (
                        <span
                          key={`pill-perf-${performer.id}`}
                          className="px-3 py-1 rounded-full text-xs font-semibold border border-indigo-400 bg-indigo-50 text-indigo-600 truncate"
                        >
                          {performer.name}
                        </span>
                      ))}
                      {selectedTags.length === 0 && selectedPerformers.length === 0 ? (
                        <span className="text-xs text-gray-400">選択なし</span>
                      ) : null}
                    </div>
                    <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={handleApplyFilters}
                          className="inline-flex items-center gap-2 rounded-md bg-rose-500 text-white px-3 py-2 text-sm font-semibold hover:bg-rose-600"
                        >
                          条件を適用
                        </button>
                        <button
                          type="button"
                          onClick={handleClearFilters}
                        className="inline-flex items-center gap-2 rounded-md border border-rose-200 bg-white px-3 py-2 text-sm font-semibold text-rose-500 hover:bg-rose-50"
                      >
                        クリア
                        </button>
                      </div>
                    </div>
                  </div>
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
                  <p className="text-xs text-gray-500 flex justify-between items-center">
                    <span>トップタグ</span>
                    <span className="text-[10px] text-gray-400">タップで条件に追加</span>
                  </p>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {topTags.length === 0 ? (
                      <span className="text-xs text-gray-400">データなし</span>
                    ) : (
                      topTags.map((tag) => {
                        const active = selectedTags.some((t) => t.id === tag.tag_id);
                        return (
                          <button
                            key={tag.tag_id}
                            type="button"
                            onClick={() => toggleTagSelection({ id: tag.tag_id, name: tag.tag_name })}
                            className={`px-2 py-1 rounded-full text-xs font-semibold border transition ${
                              active ? 'bg-rose-500 text-white border-rose-500' : 'bg-white text-gray-600 border-gray-200 hover:border-rose-300'
                            }`}
                          >
                            #{tag.tag_name}
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-500 flex justify-between items-center">
                    <span>よく観ている出演者</span>
                    <span className="text-[10px] text-gray-400">タップで条件に追加</span>
                  </p>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {topPerformers.length === 0 ? (
                      <span className="text-xs text-gray-400">データなし</span>
                    ) : (
                      topPerformers.map((performer) => {
                        const active = selectedPerformers.some((p) => p.id === performer.performer_id);
                        return (
                          <button
                            key={performer.performer_id}
                            type="button"
                            onClick={() => togglePerformerSelection({ id: performer.performer_id, name: performer.performer_name })}
                            className={`px-2 py-1 rounded-full text-xs font-semibold border transition ${
                              active ? 'bg-indigo-500 text-white border-indigo-500' : 'bg-white text-gray-600 border-gray-200 hover:border-indigo-300'
                            }`}
                          >
                            {performer.performer_name}
                          </button>
                        );
                      })
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

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Sparkles,
  Users,
  Clock,
  Wand2,
  Share2,
  Loader2,
  Check,
  Plus,
  Trash2,
  RefreshCcw,
  ListChecks,
  ArrowRight,
  Tag as TagIcon,
  Film,
  BookmarkCheck,
  ClipboardList,
  Info,
  Play,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useAiRecommend, CustomIntent, AiRecommendSection, AiRecommendSectionItem } from '@/hooks/useAiRecommend';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';
import { supabase } from '@/lib/supabase';
import { trackEvent } from '@/lib/analytics';

type ModeKey = 'focus' | 'partner' | 'quick' | 'explore' | 'relax';

type ModePreset = {
  id: ModeKey;
  label: string;
  description: string;
  rationale: string;
  tone: 'energetic' | 'warm' | 'curious' | 'calm';
  icon: React.ComponentType<{ size?: number }>;
  defaultIntent: CustomIntent;
};

type IntentOption<T extends keyof CustomIntent> = {
  value: NonNullable<CustomIntent[T]>;
  label: string;
  description: string;
};

type WatchlistEntry = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  sample_video_url: string | null;
  preview_video_url: string | null | undefined;
  product_url: string | null | undefined;
  tags: Array<{ id: string; name: string }>;
  performers: Array<{ id: string; name: string }>;
  duration_minutes: number | null;
  section_id: string;
  reason_summary: string;
  reason_detail: string;
  metrics: AiRecommendSectionItem['metrics'];
};

const MODE_PRESETS: ModePreset[] = [
  {
    id: 'focus',
    label: '集中して楽しむ',
    description: '最近のLIKE傾向にぴったり合う作品を短時間で決めたいときに。',
    rationale: '最も相性の良い候補を精査し、迷わず視聴キューを作成することを目指します。',
    tone: 'energetic',
    icon: Sparkles,
    defaultIntent: { duration: 'medium', mood: 'passion', context: 'solo' },
  },
  {
    id: 'partner',
    label: 'パートナーと',
    description: '共有しやすいテーマや柔らかいトーンの作品を組み合わせたいときに。',
    rationale: '一緒に観ても会話が弾むラインナップを中心に構成します。',
    tone: 'warm',
    icon: Users,
    defaultIntent: { duration: 'long', mood: 'sweet', context: 'partner' },
  },
  {
    id: 'quick',
    label: 'サクッと',
    description: '空き時間で素早く満足したいときに。短尺・テンポ重視。',
    rationale: '20分前後で見終えられるテンポの良い作品を優先します。',
    tone: 'energetic',
    icon: Clock,
    defaultIntent: { duration: 'short', mood: 'passion', context: 'solo' },
  },
  {
    id: 'explore',
    label: '新しい刺激',
    description: 'まだ出会っていないタグやトレンドを開拓したいときに。',
    rationale: '似た嗜好ユーザーが最近高評価した作品を中心に探索枠を組みます。',
    tone: 'curious',
    icon: Wand2,
    defaultIntent: { duration: 'medium', mood: 'curious', context: 'solo' },
  },
  {
    id: 'relax',
    label: '落ち着いて観る',
    description: '静かなトーンでゆったり観たいときのセレクション。',
    rationale: 'スローテンポで余韻を楽しめる作品と、リラックス出来る題材を揃えます。',
    tone: 'calm',
    icon: Share2,
    defaultIntent: { duration: 'long', mood: 'healing', context: 'solo' },
  },
];

const DURATION_OPTIONS: IntentOption<'duration'>[] = [
  { value: 'short', label: 'ショート', description: '〜20分' },
  { value: 'medium', label: 'スタンダード', description: '20〜45分' },
  { value: 'long', label: 'じっくり', description: '45分〜' },
];

const MOOD_OPTIONS: IntentOption<'mood'>[] = [
  { value: 'sweet', label: '甘め', description: '柔らかい雰囲気でゆったり' },
  { value: 'passion', label: '情熱的', description: '勢いと刺激重視' },
  { value: 'healing', label: '癒やし', description: '落ち着いてリラックス' },
  { value: 'curious', label: '探索', description: 'いつもと違うテーマ' },
];

const CONTEXT_OPTIONS: IntentOption<'context'>[] = [
  { value: 'solo', label: 'ひとり時間', description: 'プライベートにじっくり' },
  { value: 'partner', label: 'パートナーと', description: '共有しやすい内容で' },
  { value: 'restricted', label: '静音必須', description: '音量制限がある環境' },
];

const SWIPE_QUEUE_KEY = 'ai_rec_pending_swipe';

const formatDuration = (minutes: number | null) => {
  if (!minutes) return '—';
  if (minutes < 60) return `${minutes}分`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest === 0 ? `${hours}時間` : `${hours}時間${rest}分`;
};

const createWatchlistEntry = (
  item: AiRecommendSectionItem,
  sectionId: string,
): WatchlistEntry => ({
  id: item.id,
  title: item.title,
  thumbnail_url: item.thumbnail_url,
  sample_video_url: item.sample_video_url,
  preview_video_url: item.preview_video_url,
  product_url: item.product_url,
  tags: item.tags,
  performers: item.performers,
  duration_minutes: item.duration_minutes ?? null,
  section_id: sectionId,
  reason_summary: item.reason.summary,
  reason_detail: item.reason.detail,
  metrics: item.metrics,
});

const reorderEntries = (entries: WatchlistEntry[], fromId: string, toId: string) => {
  const current = [...entries];
  const fromIndex = current.findIndex((entry) => entry.id === fromId);
  const toIndex = current.findIndex((entry) => entry.id === toId);
  if (fromIndex === -1 || toIndex === -1 || fromIndex === toIndex) {
    return entries;
  }
  const [moved] = current.splice(fromIndex, 1);
  current.splice(toIndex, 0, moved);
  return current;
};

export default function AiRecommendPage() {
  const [modeId, setModeId] = useState<ModeKey>('focus');
  const activePreset = useMemo(
    () => MODE_PRESETS.find((preset) => preset.id === modeId) ?? MODE_PRESETS[0],
    [modeId],
  );

  const [appliedIntent, setAppliedIntent] = useState<CustomIntent>(activePreset.defaultIntent);
  const [draftIntent, setDraftIntent] = useState<CustomIntent>(activePreset.defaultIntent);

  useEffect(() => {
    setAppliedIntent(activePreset.defaultIntent);
    setDraftIntent(activePreset.defaultIntent);
  }, [activePreset]);

  const { data, loading, error, refetch } = useAiRecommend({
    modeId,
    customIntent: appliedIntent,
  });

  const { data: analysisData } = useAnalysisResults({
    windowDays: 90,
    includeNope: false,
    tagLimit: 3,
    performerLimit: 3,
    recentLimit: 0,
  });

  const [watchlist, setWatchlist] = useState<WatchlistEntry[]>([]);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [expandedItemId, setExpandedItemId] = useState<string | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [visibility, setVisibility] = useState<'private' | 'link' | 'public'>('private');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWatchlist([]);
    setNotes('');
    setVisibility('private');
  }, [modeId]);

  const handleModeChange = (preset: ModePreset) => {
    if (preset.id === modeId) return;
    setModeId(preset.id);
    trackEvent('ai_rec_mode_switch', {
      mode_id: preset.id,
      tone: preset.tone,
    });
  };

  const handleDraftIntentChange = <K extends keyof CustomIntent>(key: K, value: NonNullable<CustomIntent[K]>) => {
    setDraftIntent((prev) => ({
      ...prev,
      [key]: prev[key] === value ? prev[key] : value,
    }));
  };

  const handleApplyCustomIntent = () => {
    setAppliedIntent(draftIntent);
    toast.success('カスタムモードを適用しました');
    trackEvent('ai_rec_custom_apply', {
      mode_id: modeId,
      duration: draftIntent.duration ?? 'auto',
      mood: draftIntent.mood ?? 'auto',
      context: draftIntent.context ?? 'auto',
    });
  };

  const handleAddItem = (item: AiRecommendSectionItem, section: AiRecommendSection) => {
    setWatchlist((prev) => {
      if (prev.some((entry) => entry.id === item.id)) {
        toast('ウォッチリストに追加済みです');
        return prev;
      }
      trackEvent('ai_rec_playlist_add', {
        mode_id: modeId,
        section_id: section.id,
        video_id: item.id,
      });
      return [...prev, createWatchlistEntry(item, section.id)];
    });
  };

  const handleAddSection = (section: AiRecommendSection) => {
    setWatchlist((prev) => {
      const next = [...prev];
      section.items.forEach((item) => {
        if (!next.some((entry) => entry.id === item.id)) {
          next.push(createWatchlistEntry(item, section.id));
        }
      });
      if (next.length === prev.length) {
        toast('すべて追加済みです');
      } else {
        toast.success('セクションをウォッチリストに追加しました');
        trackEvent('ai_rec_playlist_add_section', {
          mode_id: modeId,
          section_id: section.id,
          added_count: next.length - prev.length,
        });
      }
      return next;
    });
  };

  const handleRemoveWatchlistItem = (id: string) => {
    setWatchlist((prev) => prev.filter((entry) => entry.id !== id));
  };

  const handleClearWatchlist = () => {
    setWatchlist([]);
    toast('ウォッチリストをクリアしました');
  };

  const handleDragStart = (id: string) => {
    setDraggingId(id);
  };

  const handleDragOver = (event: React.DragEvent<HTMLLIElement>, targetId: string) => {
    event.preventDefault();
    if (!draggingId || draggingId === targetId) return;
    setWatchlist((prev) => reorderEntries(prev, draggingId, targetId));
  };

  const handleDragEnd = () => {
    setDraggingId(null);
  };

  const totalDuration = useMemo(() => {
    return watchlist.reduce((total, entry) => total + (entry.duration_minutes ?? 0), 0);
  }, [watchlist]);

  const handleSendToSwipe = () => {
    if (!watchlist.length) {
      toast('ウォッチリストに作品を追加してください');
      return;
    }
    if (typeof window === 'undefined') return;
    trackEvent('ai_rec_send_to_swipe', {
      mode_id: modeId,
      count: watchlist.length,
      duration_total: totalDuration,
    });
    const payload = watchlist.map((entry, index) => ({
      id: entry.id,
      title: entry.title,
      thumbnail_url: entry.thumbnail_url,
      sample_video_url: entry.sample_video_url,
      preview_video_url: entry.preview_video_url ?? null,
      product_url: entry.product_url ?? null,
      tags: entry.tags,
      performers: entry.performers,
      reason: entry.reason_summary,
      duration_minutes: entry.duration_minutes ?? null,
      section_id: entry.section_id,
      position: index,
    }));
    window.localStorage.setItem(
      SWIPE_QUEUE_KEY,
      JSON.stringify({ items: payload, createdAt: Date.now() }),
    );
    toast.success('スワイプ画面に送信しました');
  };

  const handleSavePlaylist = useCallback(async () => {
    if (!watchlist.length) {
      toast('保存する作品がありません');
      return;
    }
    setSaving(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session?.access_token) {
        toast.error('保存するにはログインが必要です');
        setSaving(false);
        return;
      }
      const headers: HeadersInit = { Authorization: `Bearer ${session.access_token}` };
      const body = {
        mode_id: modeId,
        custom_intent: appliedIntent,
        items: watchlist.map((entry, index) => ({
          video_id: entry.id,
          position: index,
          title: entry.title ?? undefined,
          source_section: entry.section_id,
          rationale: entry.reason_summary,
          duration_minutes: entry.duration_minutes ?? null,
        })),
        notes: notes || undefined,
        visibility,
      };
      const { error: fnError } = await supabase.functions.invoke('ai-recommend-playlist', {
        headers,
        body,
      });
      if (fnError) throw fnError;
      trackEvent('ai_rec_playlist_save', {
        mode_id: modeId,
        count: watchlist.length,
        visibility,
      });
      toast.success('ウォッチリストを保存しました');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '保存に失敗しました');
    } finally {
      setSaving(false);
    }
  }, [appliedIntent, modeId, notes, visibility, watchlist]);

  const renderSection = (section: AiRecommendSection) => (
    <section key={section.id} className="rounded-2xl bg-white/85 backdrop-blur-md border border-white/60 shadow-lg p-6 flex flex-col gap-4">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <LayersIcon sectionId={section.id} />
            {section.title}
          </h3>
          <p className="text-sm text-gray-600 mt-1">{section.rationale}</p>
        </div>
        <button
          type="button"
          onClick={() => handleAddSection(section)}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-rose-500/15 text-rose-500 hover:bg-rose-500/25 transition text-sm font-semibold"
        >
          <Plus size={16} />
          すべて追加
        </button>
      </header>
      <div className="flex gap-4 overflow-x-auto pb-2">
        {section.items.map((item) => {
          const isExpanded = expandedItemId === `${section.id}:${item.id}`;
          return (
            <article
              key={item.id}
              className="w-[260px] shrink-0 rounded-2xl bg-white border border-gray-100 shadow-md flex flex-col overflow-hidden"
            >
              <div className="relative w-full aspect-[3/2] bg-gray-200">
                {item.thumbnail_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={item.thumbnail_url} alt={item.title ?? 'thumbnail'} className="absolute inset-0 w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">No Image</div>
                )}
                <div className="absolute top-2 right-2 bg-white/90 text-xs px-2 py-1 rounded-full text-gray-700">
                  {item.metrics.source === 'personalized' ? 'Personal' : item.metrics.source === 'trending' ? 'Trend' : 'Fresh'}
                </div>
              </div>
              <div className="flex flex-col gap-3 p-4 flex-1">
                <div>
                  <h4 className="text-sm font-bold text-gray-900 line-clamp-2">{item.title ?? 'タイトル未設定'}</h4>
                  <div className="mt-1 text-xs text-gray-500 flex flex-wrap gap-1">
                    {item.tags.slice(0, 2).map((tag) => (
                      <span key={tag.id} className="px-1.5 py-0.5 bg-gray-100 rounded-full border border-gray-200 text-gray-600">#{tag.name}</span>
                    ))}
                  </div>
                </div>
                <div className="space-y-1 text-xs text-gray-500">
                  <p>{item.reason.summary}</p>
                  <button
                    type="button"
                    onClick={() => {
                      const nextId = isExpanded ? null : `${section.id}:${item.id}`;
                      setExpandedItemId(nextId);
                      trackEvent('ai_rec_reason_expand', {
                        mode_id: modeId,
                        section_id: section.id,
                        video_id: item.id,
                        expanded: nextId !== null,
                      });
                    }}
                    className="inline-flex items-center gap-1 text-rose-500 hover:text-rose-400 font-semibold"
                  >
                    <Info size={14} />
                    詳細を見る
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
                    </div>
                  ) : null}
                </div>
                <div className="mt-auto flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={() => handleAddItem(item, section)}
                    className="inline-flex items-center justify-center gap-2 rounded-md bg-rose-500 text-white px-3 py-2 text-sm font-semibold hover:bg-rose-400 transition"
                  >
                    <Plus size={16} />
                    ウォッチリストに追加
                  </button>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{formatDuration(item.duration_minutes ?? null)}</span>
                    <button
                      type="button"
                      onClick={() => {
                        if (item.product_url) window.open(item.product_url, '_blank', 'noopener,noreferrer');
                      }}
                      className="inline-flex items-center gap-1 text-gray-600 hover:text-gray-800"
                    >
                      <Play size={14} />
                      作品を見る
                    </button>
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );

  const topTags = analysisData?.top_tags ?? [];
  const topPerformers = analysisData?.top_performers ?? [];

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <header className="rounded-3xl bg-white/20 backdrop-blur-xl border border-white/40 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-6 sm:px-10 py-8 text-white flex flex-col gap-4">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">AIコンシェルジュ</h1>
              <p className="text-sm sm:text-base text-white/80 mt-2">
                今日の気分と最近の嗜好から、最短ルートで視聴候補を作りましょう。スワイプと違い、まとめて比較・保存できます。
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => refetch()}
                className="inline-flex items-center gap-2 rounded-md bg-white/20 hover:bg-white/30 px-3 py-2 text-sm font-semibold transition"
              >
                <RefreshCcw size={16} />
                再生成
              </button>
              <button
                type="button"
                onClick={handleSendToSwipe}
                className="inline-flex items-center gap-2 rounded-md bg-rose-500 hover:bg-rose-400 px-3 py-2 text-sm font-semibold transition"
              >
                <ArrowRight size={16} />
                スワイプへ送る
              </button>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)_320px] gap-4">
          <aside className="flex flex-col gap-4">
            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
              <header>
                <h2 className="text-base font-bold text-gray-900">今週の嗜好スナップショット</h2>
                <p className="text-xs text-gray-500 mt-1">直近90日のLIKE履歴を概観します。</p>
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
            </section>

            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
              <header>
                <h3 className="text-base font-bold text-gray-900">モードを選ぶ</h3>
                <p className="text-xs text-gray-500 mt-1">目的に合わせてAIが提案するセットを切り替えます。</p>
              </header>
              <div className="flex flex-col gap-3">
                {MODE_PRESETS.map((preset) => {
                  const Icon = preset.icon;
                  const active = preset.id === modeId;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => handleModeChange(preset)}
                      className={`rounded-xl border px-4 py-3 text-left transition ${
                        active
                          ? 'border-rose-400 bg-rose-500/10 text-rose-600 shadow-sm'
                          : 'border-gray-200 hover:border-rose-200 hover:bg-rose-50 text-gray-700'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-1">
                          <Icon size={18} />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold">{preset.label}</span>
                            {active ? (
                              <span className="inline-flex items-center gap-1 text-xs text-rose-500">
                                <Check size={14} />
                                選択中
                              </span>
                            ) : null}
                          </div>
                          <p className="text-xs text-gray-500 mt-1">{preset.description}</p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>

            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
              <header>
                <h3 className="text-base font-bold text-gray-900">カスタムモード</h3>
                <p className="text-xs text-gray-500 mt-1">気分に合わせて細かくチューニングし、AIに再提案させます。</p>
              </header>
              <div className="flex flex-col gap-3 text-sm">
                <IntentSelector
                  label="視聴時間"
                  options={DURATION_OPTIONS}
                  current={draftIntent.duration ?? null}
                  onSelect={(value) => handleDraftIntentChange('duration', value)}
                />
                <IntentSelector
                  label="雰囲気"
                  options={MOOD_OPTIONS}
                  current={draftIntent.mood ?? null}
                  onSelect={(value) => handleDraftIntentChange('mood', value)}
                />
                <IntentSelector
                  label="視聴環境"
                  options={CONTEXT_OPTIONS}
                  current={draftIntent.context ?? null}
                  onSelect={(value) => handleDraftIntentChange('context', value)}
                />
              </div>
              <button
                type="button"
                onClick={handleApplyCustomIntent}
                className="inline-flex items-center justify-center gap-2 rounded-md bg-gray-900 text-white px-3 py-2 text-sm font-semibold hover:bg-gray-800 transition"
              >
                <Wand2 size={16} />
                AIに依頼
              </button>
              <p className="text-[11px] text-gray-500">
                適用すると画面中央の提案が更新されます。意図が合わない場合は別のモードを選び直してください。
              </p>
            </section>
          </aside>

          <div className="flex flex-col gap-4">
            {loading ? (
              <div className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-12 flex flex-col items-center justify-center gap-3 text-gray-600">
                <Loader2 className="animate-spin" size={28} />
                <p className="text-sm">AIが最適な組み合わせを考えています…</p>
              </div>
            ) : null}
            {error ? (
              <div className="rounded-2xl bg-red-100 border border-red-200 text-red-700 p-4 text-sm">
                {error}
              </div>
            ) : null}
            {data?.sections?.length
              ? data.sections.map((section) => renderSection(section))
              : (!loading && !error) ? (
                <div className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-10 text-center text-gray-500">
                  提案できる候補が見つかりませんでした。モードやカスタム設定を見直してください。
                </div>
              ) : null}
          </div>

          <aside className="flex flex-col gap-4">
            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
              <header className="flex items-center justify-between gap-2">
                <div>
                  <h3 className="text-base font-bold text-gray-900">ウォッチリスト</h3>
                  <p className="text-xs text-gray-500 mt-1">決めた作品を並び替えて保存・共有できます。</p>
                </div>
                <button
                  type="button"
                  onClick={handleClearWatchlist}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  クリア
                </button>
              </header>
              <div className="rounded-xl bg-gray-100 border border-gray-200 p-3 text-xs text-gray-600 flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span>作品数</span>
                  <span className="font-semibold text-gray-800">{watchlist.length} 件</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>想定視聴時間</span>
                  <span className="font-semibold text-gray-800">{formatDuration(totalDuration)}</span>
                </div>
              </div>
              <div className="flex flex-col gap-3 max-h-[360px] overflow-y-auto pr-1">
                {watchlist.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-gray-300 p-6 text-center text-xs text-gray-500">
                    まだウォッチリストに作品がありません。中央のカードから「追加」を押してください。
                  </div>
                ) : (
                  <ul className="flex flex-col gap-3">
                    {watchlist.map((entry, index) => (
                      <li
                        key={entry.id}
                        draggable
                        onDragStart={() => handleDragStart(entry.id)}
                        onDragOver={(event) => handleDragOver(event, entry.id)}
                        onDragEnd={handleDragEnd}
                        className={`rounded-xl border bg-white px-3 py-3 text-gray-800 shadow-sm ${draggingId === entry.id ? 'opacity-70' : ''}`}
                      >
                        <div className="flex items-start gap-3">
                          <div className="flex flex-col items-center text-xs text-gray-500">
                            <span>{index + 1}</span>
                            <span className="mt-1 cursor-grab select-none">⋮⋮</span>
                          </div>
                          <div className="flex-1 flex flex-col gap-2">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <p className="text-sm font-semibold line-clamp-2">{entry.title ?? 'タイトル未設定'}</p>
                                <p className="text-[11px] text-gray-500 mt-1">{entry.reason_summary}</p>
                              </div>
                              <button
                                type="button"
                                onClick={() => handleRemoveWatchlistItem(entry.id)}
                                className="text-gray-400 hover:text-rose-500 transition"
                              >
                                <Trash2 size={16} />
                              </button>
                            </div>
                            <div className="text-[11px] text-gray-500 flex flex-wrap gap-2">
                              {entry.tags.slice(0, 2).map((tag) => (
                                <span key={tag.id} className="inline-flex items-center gap-1">
                                  <TagIcon size={11} />
                                  {tag.name}
                                </span>
                              ))}
                              {entry.duration_minutes ? (
                                <span className="inline-flex items-center gap-1">
                                  <Clock size={11} />
                                  {formatDuration(entry.duration_minutes)}
                                </span>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </section>

            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
              <header>
                <h3 className="text-base font-bold text-gray-900">保存と共有</h3>
                <p className="text-xs text-gray-500 mt-1">AI提案セットを保存して、後から呼び出したり共有できます。</p>
              </header>
              <div className="flex flex-col gap-2">
                <label className="flex flex-col gap-1 text-xs text-gray-600">
                  メモ（任意）
                  <textarea
                    value={notes}
                    onChange={(event) => setNotes(event.target.value)}
                    rows={3}
                    className="rounded-md border border-gray-200 px-2 py-1 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-rose-200"
                    placeholder="セッションの目的や共有メモを記録できます"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-gray-600">
                  公開設定
                  <select
                    value={visibility}
                    onChange={(event) => setVisibility(event.target.value as typeof visibility)}
                    className="rounded-md border border-gray-200 px-2 py-1 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-rose-200"
                  >
                    <option value="private">非公開（自分のみ）</option>
                    <option value="link">リンク共有</option>
                    <option value="public">公開</option>
                  </select>
                </label>
              </div>
              <button
                type="button"
                onClick={handleSavePlaylist}
                disabled={saving || !watchlist.length}
                className={`inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-semibold transition ${
                  watchlist.length
                    ? 'bg-gray-900 text-white hover:bg-gray-800'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }`}
              >
                {saving ? <Loader2 size={16} className="animate-spin" /> : <BookmarkCheck size={16} />}
                保存する
              </button>
              <button
                type="button"
                onClick={handleSendToSwipe}
                className="inline-flex items-center justify-center gap-2 rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:border-gray-400 hover:text-gray-900 transition"
              >
                <ClipboardList size={16} />
                スワイプ開始キューに送る
              </button>
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}

function IntentSelector<T extends keyof CustomIntent>({
  label,
  options,
  current,
  onSelect,
}: {
  label: string;
  options: IntentOption<T>[];
  current: CustomIntent[T] | null;
  onSelect: (value: IntentOption<T>['value']) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-semibold text-gray-500">{label}</p>
      <div className="grid grid-cols-1 gap-2">
        {options.map((option) => {
          const active = option.value === current;
          return (
            <button
              type="button"
              key={option.value}
              onClick={() => onSelect(option.value)}
              className={`rounded-lg border px-3 py-2 text-left transition ${
                active
                  ? 'border-rose-400 bg-rose-500/10 text-rose-600 shadow-sm'
                  : 'border-gray-200 hover:border-rose-200 hover:bg-rose-50 text-gray-700'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold">{option.label}</span>
                {active ? <Check size={16} className="text-rose-500" /> : null}
              </div>
              <p className="text-xs text-gray-500 mt-1">{option.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LayersIcon({ sectionId }: { sectionId: string }) {
  if (sectionId.includes('trend')) return <FlameIcon />;
  if (sectionId.includes('fresh')) return <Sparkles size={18} className="text-rose-400" />;
  if (sectionId.includes('community')) return <Users size={18} className="text-indigo-400" />;
  return <ListChecks size={18} className="text-gray-500" />;
}

function FlameIcon() {
  return <Film size={18} className="text-orange-400" />;
}

'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Sparkles,
  Users,
  Clock,
  Wand2,
  Loader2,
  Check,
  RefreshCcw,
  Info,
  Play,
  Flame,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useAiRecommend, CustomIntent, AiRecommendSection } from '@/hooks/useAiRecommend';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';
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

const MODE_PRESETS: ModePreset[] = [
  {
    id: 'focus',
    label: '集中して楽しむ',
    description: '最近のLIKE傾向に寄せた、最短で満足できる作品を中心に提供します。',
    rationale: '適合度と類似タグを重視し、迷わず視聴に移れるように構成します。',
    tone: 'energetic',
    icon: Sparkles,
    defaultIntent: { duration: 'medium', mood: 'passion', context: 'solo' },
  },
  {
    id: 'partner',
    label: 'パートナーと',
    description: '共有しやすいテーマや柔らかいトーンの作品を組み合わせます。',
    rationale: 'パートナーと観ても会話が弾みやすいラインナップをバランスよく選定します。',
    tone: 'warm',
    icon: Users,
    defaultIntent: { duration: 'long', mood: 'sweet', context: 'partner' },
  },
  {
    id: 'quick',
    label: 'サクッと',
    description: '空き時間で素早く楽しめる短尺・テンポ重視の候補を集めます。',
    rationale: '視聴所要時間の短さとテンポ感に重点を置きます。',
    tone: 'energetic',
    icon: Clock,
    defaultIntent: { duration: 'short', mood: 'passion', context: 'solo' },
  },
  {
    id: 'explore',
    label: '新しい刺激',
    description: 'まだ出会っていないタグや出演者を中心に、探索的なラインナップを提示します。',
    rationale: '似た嗜好ユーザーの最新評価や人気の揺れ動きを取り込みます。',
    tone: 'curious',
    icon: Wand2,
    defaultIntent: { duration: 'medium', mood: 'curious', context: 'solo' },
  },
  {
    id: 'relax',
    label: '落ち着いて観る',
    description: '静かなトーンでじっくり観られる作品をピックアップします。',
    rationale: '穏やかなテンポや癒やし系のタグを優先し、リラックスできる構成にします。',
    tone: 'calm',
    icon: Users,
    defaultIntent: { duration: 'long', mood: 'healing', context: 'solo' },
  },
];

const DURATION_OPTIONS: IntentOption<'duration'>[] = [
  { value: 'short', label: 'ショート', description: '〜20分程度で完結' },
  { value: 'medium', label: 'スタンダード', description: '20〜45分で満足' },
  { value: 'long', label: 'じっくり', description: '45分以上ゆっくり楽しむ' },
];

const MOOD_OPTIONS: IntentOption<'mood'>[] = [
  { value: 'sweet', label: '甘め', description: '柔らかく穏やかな雰囲気' },
  { value: 'passion', label: '情熱的', description: 'テンポが良く刺激的' },
  { value: 'healing', label: '癒やし', description: 'リラックスして観たい' },
  { value: 'curious', label: '探索', description: 'いつもと違う刺激にチャレンジ' },
];

const CONTEXT_OPTIONS: IntentOption<'context'>[] = [
  { value: 'solo', label: 'ひとり時間', description: 'プライベートに没入' },
  { value: 'partner', label: 'パートナーと', description: '共有しながら楽しむ' },
  { value: 'restricted', label: '静音必須', description: '音量制限がある環境向け' },
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
  const [modeId, setModeId] = useState<ModeKey>('focus');
  const activePreset = useMemo(
    () => MODE_PRESETS.find((preset) => preset.id === modeId) ?? MODE_PRESETS[0],
    [modeId],
  );
  const [appliedIntent, setAppliedIntent] = useState<CustomIntent>(activePreset.defaultIntent);
  const [draftIntent, setDraftIntent] = useState<CustomIntent>(activePreset.defaultIntent);
  const [expandedItemId, setExpandedItemId] = useState<string | null>(null);

  useEffect(() => {
    setAppliedIntent(activePreset.defaultIntent);
    setDraftIntent(activePreset.defaultIntent);
    setExpandedItemId(null);
  }, [activePreset]);

  const { data, loading, error, refetch } = useAiRecommend({
    modeId,
    customIntent: appliedIntent,
    limitPerSection: 8,
  });

  const { data: analysisData } = useAnalysisResults({
    windowDays: 90,
    includeNope: false,
    tagLimit: 3,
    performerLimit: 3,
    recentLimit: 0,
  });

  const handleModeChange = (preset: ModePreset) => {
    if (preset.id === modeId) return;
    setModeId(preset.id);
    setExpandedItemId(null);
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
    setExpandedItemId(null);
    toast.success('カスタム設定を適用しました');
    trackEvent('ai_rec_custom_apply', {
      mode_id: modeId,
      duration: draftIntent.duration ?? 'auto',
      mood: draftIntent.mood ?? 'auto',
      context: draftIntent.context ?? 'auto',
    });
  };

  const topTags = analysisData?.top_tags ?? [];
  const topPerformers = analysisData?.top_performers ?? [];

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
      <div className="flex gap-4 overflow-x-auto pb-2">
        {section.items.map((item) => {
          const composedId = `${section.id}:${item.id}`;
          const isExpanded = expandedItemId === composedId;
          return (
            <article
              key={item.id}
              className="w-[280px] shrink-0 rounded-2xl bg-white border border-gray-100 shadow-md flex flex-col overflow-hidden"
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
                    {item.tags.slice(0, 3).map((tag) => (
                      <span key={tag.id} className="px-1.5 py-0.5 bg-gray-100 rounded-full border border-gray-200 text-gray-600">
                        #{tag.name}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="space-y-1 text-xs text-gray-500">
                  <p>{item.reason.summary}</p>
                  <button
                    type="button"
                    onClick={() => {
                      const nextId = isExpanded ? null : composedId;
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
                      {item.metrics.popularity_score ? (
                        <p>人気指標 {item.metrics.popularity_score.toLocaleString('ja-JP')}</p>
                      ) : null}
                      {item.metrics.product_released_at ? (
                        <p>リリース日 {formatDate(item.metrics.product_released_at)}</p>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                <div className="mt-auto flex items-center justify-between text-xs text-gray-500">
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

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <section className="w-full rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 flex flex-col gap-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-white">
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">AIで探す</h1>
              <p className="text-sm sm:text-base text-white/80 mt-2">
                嗜好と気分に合わせてAIがまとめた候補セットを提示します。スワイプで評価に入る前に、どのラインを深掘りするかをここで決めましょう。
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
              再生成
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-4">
            <aside className="flex flex-col gap-4">
              <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
                <header>
                  <h2 className="text-base font-bold text-gray-900">最新の嗜好スナップショット</h2>
                  <p className="text-xs text-gray-500 mt-1">直近90日のLIKE履歴から抽出した傾向です。</p>
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

              <section className="rounded-2xl.bg-white/85 backdrop-blur border border-white/60 shadow-lg p-5 flex flex-col gap-4">
                <header>
                  <h3 className="text-base font-bold text-gray-900">カスタムモード</h3>
                  <p className="text-xs text-gray-500.mt-1">気分に合わせて細かくチューニングし、AIに再提案させます。</p>
                </header>
                <div className="flex flex-col gap-3 text-sm">
                  <IntentSelector
                    label="視聴時間の目安"
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
                  適用すると右側の提案が更新されます。結果が合わない場合はモードを切り替えて再度試してください。
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
                    提案できる候補が見つかりませんでした。モードやカスタム設定を見直して再生成してください。
                  </div>
                ) : null}
            </div>
          </div>
        </section>
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
  if (sectionId.includes('trend')) return <Flame size={18} className="text-orange-400" />;
  if (sectionId.includes('fresh')) return <Sparkles size={18} className="text-rose-400" />;
  if (sectionId.includes('community')) return <Users size={18} className="text-indigo-400" />;
  return <Clock size={18} className="text-gray-500" />;
}

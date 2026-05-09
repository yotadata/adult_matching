'use client';

import { useMemo, useRef, useState } from 'react';
import { Share2, Tag as TagIcon, Users, Clock, ExternalLink } from 'lucide-react';
import { toPng } from 'html-to-image';
import ShareModal from '@/components/ShareModal';
import AnalysisShareCard from '@/components/AnalysisShareCard';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';
import type { AnalysisTag, AnalysisPerformer } from '@/hooks/useAnalysisResults';

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
    return new Intl.DateTimeFormat('ja-JP', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);
  } catch { return '—'; }
};

const formatCount = (value: number): string => value.toLocaleString('ja-JP');

// ── タイプ判定ロジック（AnalysisShareCardと共通） ─────────────────
type TypeProfile = { keywords: string[]; typeName: string; highlight: string; headerCopy: string };

const TYPE_PROFILES: TypeProfile[] = [
  { keywords: ['美少女', 'ロリ', '制服', '女子校生', '妹'], typeName: '清楚ロリ系', highlight: '透明感あるかわいいムードでときめきがち。', headerCopy: 'あどけない雰囲気の作品を追いかける王道ロリ派。' },
  { keywords: ['巨乳', '爆乳', 'お姉さん', 'むちむち', '痴女'], typeName: '巨乳お姉さん系', highlight: '包容力たっぷりの大人なお姉さんでとろけたい。', headerCopy: '余裕のあるお姉さんに甘やかされたい欲が爆発中。' },
  { keywords: ['人妻', '不倫', '寝取られ', 'NTR', '禁断'], typeName: '背徳NTR系', highlight: '禁断のストーリーで心拍を上げるのが快感。', headerCopy: '日常では味わえない背徳展開に特大の刺さり方。' },
  { keywords: ['ギャル', '日焼け', 'ビッチ', 'パリピ'], typeName: 'ギラギラギャル系', highlight: '勢いとノリで押してくるギャルに弱い。', headerCopy: '明るいギャルエネルギーでテンションを上げるタイプ。' },
  { keywords: ['単体作品', '王道', 'ノーマル'], typeName: '王道単体派', highlight: '企画よりもシンプルな単体作で集中したい。', headerCopy: 'じっくり堪能できる王道スタイルを求める派。' },
  { keywords: ['SM', '拘束', '調教', '支配', '主従'], typeName: '刺激スパイス系', highlight: 'ピリ辛な刺激で一気にスイッチが入る。', headerCopy: '刺激的なシチュをアクセントにしたい攻め志向。' },
];

const DEFAULT_PROFILE = { typeName: 'バランス型', highlight: 'ジャンルを渡り歩きつつ、その日の気分で決める柔軟派。', headerCopy: '王道からマニアックまで幅広くいける万能スタイル。' };

function deriveTypeProfile(tags: AnalysisTag[]) {
  const tagNames = tags.map((t) => t.tag_name);
  return TYPE_PROFILES.find((p) => p.keywords.some((kw) => tagNames.some((n) => n.includes(kw)))) ?? DEFAULT_PROFILE;
}

// 変態度スコア（like_ratio・総判断数・タグ多様性から算出）
function calcHentaiScore(likeRatio: number | null, sampleSize: number, tagCount: number): number {
  const ratio = likeRatio ?? 0;
  // like_ratioが極端（高すぎ or 低すぎ）ほど変態度が高い
  const ratioPart = Math.abs(ratio - 0.4) * 1.5; // 0.4が平均的、そこからの乖離
  // 判断数が多いほど変態度が高い（上限あり）
  const activityPart = Math.min(sampleSize / 500, 1) * 0.3;
  // タグ多様性（少ないほどこだわり強い→変態度高い）
  const diversityPart = tagCount <= 2 ? 0.2 : tagCount <= 5 ? 0.1 : 0;
  const raw = 0.35 + ratioPart + activityPart + diversityPart;
  return Math.round(Math.min(Math.max(raw, 0.2), 0.99) * 100);
}

const RANK_COLORS = ['from-yellow-400 to-amber-500', 'from-slate-300 to-slate-400', 'from-amber-600 to-amber-700'];

function RankBadge({ rank }: { rank: number }) {
  const color = RANK_COLORS[rank - 1] ?? 'from-rose-400 to-fuchsia-500';
  return (
    <span className={`w-7 h-7 rounded-full bg-gradient-to-br ${color} text-white text-xs font-black flex items-center justify-center shrink-0 shadow`}>
      {rank}
    </span>
  );
}

function TagRankItem({ tag, rank }: { tag: AnalysisTag; rank: number }) {
  const ratio = tag.like_ratio ?? 0;
  return (
    <div className="flex items-center gap-3 rounded-xl bg-white/70 border border-white/60 shadow-sm px-4 py-3">
      <RankBadge rank={rank} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-2 mb-1">
          <span className="text-sm font-bold text-gray-900 truncate">#{tag.tag_name}</span>
          <span className="text-sm font-black text-rose-500 shrink-0">{formatPercent(tag.like_ratio)}</span>
        </div>
        <div className="h-1.5 bg-rose-50 rounded-full overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-rose-400 to-fuchsia-500 transition-all" style={{ width: `${Math.round(ratio * 100)}%` }} />
        </div>
        <div className="flex justify-between mt-1 text-[10px] text-gray-400">
          <span>気になる {formatCount(tag.likes)}件</span>
          {tag.share != null && <span>シェア {formatPercent(tag.share)}</span>}
        </div>
      </div>
      {tag.representative_video?.product_url && (
        <a href={tag.representative_video.product_url} target="_blank" rel="noopener noreferrer" className="shrink-0 text-gray-400 hover:text-rose-400 transition">
          <ExternalLink size={14} />
        </a>
      )}
    </div>
  );
}

function PerformerRankItem({ performer, rank }: { performer: AnalysisPerformer; rank: number }) {
  return (
    <div className="flex items-center gap-3 rounded-xl bg-white/70 border border-white/60 shadow-sm px-4 py-3">
      <RankBadge rank={rank} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-sm font-bold text-gray-900 truncate">{performer.performer_name}</span>
          <span className="text-sm font-black text-indigo-500 shrink-0">{formatPercent(performer.like_ratio)}</span>
        </div>
        <div className="text-[10px] text-gray-400 mt-0.5">
          気になる {formatCount(performer.likes)}件
          {performer.share != null && <span className="ml-2">シェア {formatPercent(performer.share)}</span>}
        </div>
      </div>
      {performer.representative_video?.product_url && (
        <a href={performer.representative_video.product_url} target="_blank" rel="noopener noreferrer" className="shrink-0 text-gray-400 hover:text-indigo-400 transition">
          <ExternalLink size={14} />
        </a>
      )}
    </div>
  );
}

// ── メインページ ──────────────────────────────────────────────────
export default function InsightsPage() {
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
    return summary.window_days === null ? '全期間' : `直近 ${summary.window_days} 日`;
  }, [summary]);

  const typeProfile = useMemo(() => deriveTypeProfile(topTags), [topTags]);

  const hentaiScore = useMemo(() => {
    if (!summary) return null;
    return calcHentaiScore(summary.like_ratio, summary.sample_size, topTags.length);
  }, [summary, topTags]);

  const shareUrl = typeof window !== 'undefined' ? window.location.href : 'https://seiheki.me/insights';
  const primaryTag = topTags[0];
  const primaryPerformer = topPerformers[0];
  const shareTagNames = topTags.slice(0, 2).map((t) => `#${t.tag_name}`);
  const shareText = (() => {
    if (primaryTag && primaryPerformer) return `最近は ${primaryPerformer.performer_name} が出る #${primaryTag.tag_name} 系で毎回「気になる」。あなたも #あなたの性癖 を診断して抜けるポイントをシェアしよう。`;
    if (shareTagNames.length > 0) return `今の抜けるタグは ${shareTagNames.join(' / ')}。あなたも #あなたの性癖 を診断して好みカードを作ろう。`;
    if (primaryPerformer) return `${primaryPerformer.performer_name} が出ていたら即「気になる」。あなたも #あなたの性癖 を診断してみよう。`;
    return `${windowLabel}の好み分析結果をシェアしました。あなたも #あなたの性癖 を診断してみて。`;
  })();

  const handleShare = async () => {
    if (!cardRef.current || !summary) { window.alert('データを取得してからシェアしてください。'); return; }
    try {
      const dataUrl = await toPng(cardRef.current, { cacheBust: true, pixelRatio: 2, backgroundColor: '#050113' });
      setShareImageUrl(dataUrl);
      setIsShareModalOpen(true);
    } catch {
      if (navigator.share) {
        try { await navigator.share({ title: 'あなたの性癖分析結果', text: shareText, url: shareUrl }); } catch { /* noop */ }
      }
    }
  };

  return (
    <>
      <main className="w-full min-h-screen px-0 sm:px-4 py-8">
        <div className="max-w-7xl mx-auto flex flex-col gap-5">

          {/* ── ヒーロー：タイプ名・変態度 ── */}
          <section className="w-full rounded-2xl overflow-hidden relative" style={{ background: 'linear-gradient(135deg, #1a0a2e 0%, #2d0a1f 50%, #0f1a3d 100%)' }}>
            <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'radial-gradient(circle at 20% 50%, #f43f5e 0%, transparent 50%), radial-gradient(circle at 80% 20%, #8b5cf6 0%, transparent 50%)' }} />
            <div className="relative px-6 py-8 sm:px-10 sm:py-10 flex flex-col sm:flex-row sm:items-end gap-6">
              <div className="flex-1">
                <p className="text-xs font-bold tracking-[0.4em] text-rose-400 uppercase mb-2">あなたの性癖タイプ</p>
                {loading ? (
                  <div className="h-10 w-48 bg-white/10 rounded-lg animate-pulse" />
                ) : (
                  <h1 className="text-3xl sm:text-4xl font-black text-white leading-tight">{typeProfile.typeName}</h1>
                )}
                <p className="text-sm text-white/70 mt-2 leading-relaxed">{loading ? '' : typeProfile.headerCopy}</p>
              </div>

              {/* 変態度スコア */}
              <div className="shrink-0 flex flex-col items-center gap-1">
                <p className="text-[10px] font-bold tracking-[0.3em] text-white/50 uppercase">変態度</p>
                <div className="relative w-24 h-24">
                  <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                    <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="10" />
                    <circle
                      cx="50" cy="50" r="42" fill="none"
                      stroke="url(#scoreGrad)" strokeWidth="10"
                      strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 42}`}
                      strokeDashoffset={loading || hentaiScore === null ? 2 * Math.PI * 42 : 2 * Math.PI * 42 * (1 - hentaiScore / 100)}
                      style={{ transition: 'stroke-dashoffset 1s ease' }}
                    />
                    <defs>
                      <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f43f5e" />
                        <stop offset="100%" stopColor="#a855f7" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-black text-white">
                      {loading || hentaiScore === null ? '—' : `${hentaiScore}`}
                    </span>
                  </div>
                </div>
                <p className="text-[10px] text-white/40">/ 100</p>
              </div>

              {/* シェアボタン */}
              <button
                onClick={handleShare}
                className="shrink-0 inline-flex items-center gap-2 rounded-full bg-white/15 hover:bg-white/25 border border-white/20 px-4 py-2 text-sm font-semibold text-white transition backdrop-blur"
              >
                <Share2 size={15} />
                シェア
              </button>
            </div>

            {/* 期間切替 */}
            <div className="relative px-6 pb-5 sm:px-10 flex flex-wrap gap-2">
              {WINDOW_OPTIONS.map((opt) => {
                const active = opt.value === windowDays || (opt.value === null && windowDays === null);
                return (
                  <button
                    key={opt.label}
                    onClick={() => setWindowDays(opt.value)}
                    className={`px-3 py-1 rounded-full text-xs font-semibold transition ${active ? 'bg-white text-rose-500 shadow' : 'bg-white/10 text-white/70 hover:bg-white/20'}`}
                  >
                    {opt.label}
                  </button>
                );
              })}
              {summary && (
                <span className="ml-auto text-xs text-white/40 self-center">
                  判断数 {formatCount(summary.sample_size)}件
                </span>
              )}
            </div>
          </section>

          {error && (
            <div className="rounded-2xl bg-red-500/10 border border-red-400/40 text-red-400 px-4 py-3 text-sm">{error}</div>
          )}

          {/* ── サマリー数値 ── */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: '気になる', value: summary ? formatCount(summary.total_likes) : '—', sub: windowLabel, color: 'text-rose-400' },
              { label: 'スキップ', value: summary ? formatCount(summary.total_nope) : '—', sub: windowLabel, color: 'text-slate-400' },
              { label: '気になる率', value: summary ? formatPercent(summary.like_ratio) : '—', sub: `${formatCount(summary?.sample_size ?? 0)}件中`, color: 'text-fuchsia-400' },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl bg-white/15 backdrop-blur border border-white/20 px-4 py-4 text-center">
                <p className="text-[11px] text-white/50 mb-1">{item.label}</p>
                {loading ? (
                  <div className="h-7 w-16 mx-auto bg-white/10 rounded animate-pulse" />
                ) : (
                  <p className={`text-2xl font-black ${item.color}`}>{item.value}</p>
                )}
                <p className="text-[10px] text-white/30 mt-0.5">{item.sub}</p>
              </div>
            ))}
          </div>

          {/* ── タグ・出演者ランキング ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/50 shadow-lg p-5 text-gray-900 flex flex-col gap-3">
              <header className="flex items-center gap-2 mb-1">
                <TagIcon size={17} className="text-rose-400" />
                <h2 className="text-base font-bold">好きなタグ Top {topTags.length}</h2>
              </header>
              {loading ? (
                Array.from({ length: 3 }).map((_, i) => <div key={i} className="h-16 rounded-xl bg-gray-100 animate-pulse" />)
              ) : topTags.length === 0 ? (
                <p className="text-sm text-gray-400 py-4 text-center">「気になる」を付けるとここに表示されます</p>
              ) : (
                topTags.map((tag, i) => <TagRankItem key={tag.tag_id} tag={tag} rank={i + 1} />)
              )}
            </section>

            <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/50 shadow-lg p-5 text-gray-900 flex flex-col gap-3">
              <header className="flex items-center gap-2 mb-1">
                <Users size={17} className="text-indigo-400" />
                <h2 className="text-base font-bold">よく見る出演者 Top {topPerformers.length}</h2>
              </header>
              {loading ? (
                Array.from({ length: 3 }).map((_, i) => <div key={i} className="h-14 rounded-xl bg-gray-100 animate-pulse" />)
              ) : topPerformers.length === 0 ? (
                <p className="text-sm text-gray-400 py-4 text-center">「気になる」とした作品の出演者がここに表示されます</p>
              ) : (
                topPerformers.map((perf, i) => <PerformerRankItem key={perf.performer_id} performer={perf} rank={i + 1} />)
              )}
            </section>
          </div>

          {/* ── 直近の判断履歴 ── */}
          <section className="rounded-2xl bg-white/85 backdrop-blur border border-white/50 shadow-lg p-5 text-gray-900">
            <header className="flex items-center gap-2 mb-4">
              <Clock size={17} className="text-rose-400" />
              <h2 className="text-base font-bold">直近の判断履歴</h2>
            </header>
            {loading ? (
              <div className="space-y-2">{Array.from({ length: 4 }).map((_, i) => <div key={i} className="h-16 rounded-xl bg-gray-100 animate-pulse" />)}</div>
            ) : recent.length === 0 ? (
              <p className="text-sm text-gray-400 py-4 text-center">まだ判断履歴がありません</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full table-fixed border-separate border-spacing-y-2 text-sm text-gray-700">
                  <thead>
                    <tr className="text-xs uppercase tracking-wide text-gray-400">
                      <th className="px-3 py-2 text-left font-semibold w-[35%]">作品</th>
                      <th className="px-3 py-2 text-left font-semibold w-[12%]">判断</th>
                      <th className="px-3 py-2 text-left font-semibold w-[22%]">タグ</th>
                      <th className="px-3 py-2 text-left font-semibold w-[18%]">出演者</th>
                      <th className="px-3 py-2 text-left font-semibold w-[13%]">日時</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((item) => (
                      <tr key={`${item.video_id}-${item.decided_at}`} className="bg-white/80 border border-gray-200/80 rounded-xl align-top">
                        <td className="px-3 py-3 align-top">
                          <div className="flex items-start gap-2">
                            <div className="h-12 w-16 shrink-0 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
                              {item.thumbnail_url && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={item.thumbnail_url} alt={item.title ?? ''} className="w-full h-full object-cover" />
                              )}
                            </div>
                            <span className="text-xs font-semibold text-gray-900 line-clamp-3 leading-snug">{item.title ?? '—'}</span>
                          </div>
                        </td>
                        <td className="px-3 py-3 align-top">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold ${item.decision_type === 'like' ? 'bg-rose-100 text-rose-500' : 'bg-gray-100 text-gray-500'}`}>
                            {item.decision_type === 'like' ? '💗 気になる' : 'スキップ'}
                          </span>
                        </td>
                        <td className="px-3 py-3 align-top">
                          <div className="flex flex-wrap gap-1">
                            {item.tags.slice(0, 3).map((tag) => (
                              <span key={tag.id} className="px-1.5 py-0.5 rounded-full bg-rose-50 text-rose-500 border border-rose-100 text-[10px]">#{tag.name}</span>
                            ))}
                          </div>
                        </td>
                        <td className="px-3 py-3 align-top">
                          <div className="flex flex-wrap gap-1">
                            {item.performers.slice(0, 2).map((p) => (
                              <span key={p.id} className="px-1.5 py-0.5 rounded-full bg-indigo-50 text-indigo-500 border border-indigo-100 text-[10px]">{p.name}</span>
                            ))}
                          </div>
                        </td>
                        <td className="px-3 py-3 align-top text-xs text-gray-400">{formatDate(item.decided_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* ── 集計メモ ── */}
          {summary && (
            <div className="rounded-2xl bg-white/10 border border-white/20 px-5 py-4 text-xs text-white/40 flex flex-wrap gap-4">
              <span>集計対象: {windowLabel}（最大 {MAX_FETCH.toLocaleString('ja-JP')} 件）</span>
              <span>最初の判断: {formatDate(summary.first_decision_at)}</span>
              <span>最新の判断: {formatDate(summary.latest_decision_at)}</span>
            </div>
          )}
        </div>
      </main>

      <ShareModal
        isOpen={isShareModalOpen}
        onClose={() => setIsShareModalOpen(false)}
        imageUrl={shareImageUrl}
        shareUrl={shareUrl}
        shareText={shareText}
      />
      <div className="absolute -left-[9999px] top-0 pointer-events-none select-none" aria-hidden>
        <AnalysisShareCard ref={cardRef} summary={summary} topTags={topTags} topPerformers={topPerformers} shareUrl={SHARE_CARD_LANDING_URL} />
      </div>
    </>
  );
}

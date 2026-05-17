'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Search, Heart, Play, X, Sparkles, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { Combobox } from '@headlessui/react';
import { useAiRecommend, type AiRecommendSection } from '@/hooks/useAiRecommend';
import { useVideoSearch, type VideoSearchItem } from '@/hooks/useVideoSearch';
import { useAnalysisResults } from '@/hooks/useAnalysisResults';
import { supabase } from '@/lib/supabase';
import { isUpcomingRelease } from '@/lib/videoMeta';

const AF_ID = 'yotadata2-001';
const toAffiliateUrl = (raw?: string | null) => {
  if (!raw) return '#';
  try {
    if (raw.startsWith('https://al.fanza.co.jp/')) {
      const url = new URL(raw);
      url.searchParams.set('af_id', AF_ID);
      return url.toString();
    }
  } catch { /* noop */ }
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
};

type TagOrPerformer = { id: string; name: string };

// ── 動画カード ───────────────────────────────────────────
function VideoCard({
  id,
  title,
  thumbnail_url,
  product_url,
  product_released_at,
  tags,
  aiReason,
  aiSource,
  isLiked,
  isLiking,
  onLike,
}: {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  product_url: string | null;
  product_released_at: string | null;
  tags: Array<{ id: string; name: string }>;
  aiReason?: string;
  aiSource?: string;
  isLiked: boolean;
  isLiking: boolean;
  onLike: () => void;
}) {
  const upcoming = isUpcomingRelease(product_released_at);
  const href = toAffiliateUrl(product_url);

  return (
    <article className="rounded-xl bg-white border border-gray-100 shadow-sm flex flex-col overflow-hidden text-sm">
      <div className="relative w-full aspect-[3/2] bg-gray-200 flex-shrink-0">
        {thumbnail_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumbnail_url} alt={title ?? 'thumbnail'} className="absolute inset-0 w-full h-full object-cover" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs">No Image</div>
        )}
        {upcoming && (
          <div className="absolute top-2 left-2 bg-amber-100/90 text-[10px] px-1.5 py-0.5 rounded-full text-amber-800 border border-amber-200">
            予約作品
          </div>
        )}
        {aiSource && (
          <div className="absolute bottom-2 right-2 bg-black/60 text-[10px] px-1.5 py-0.5 rounded-full text-white flex items-center gap-0.5">
            <Sparkles size={9} />
            {sourceLabel(aiSource)}
          </div>
        )}
      </div>

      <div className="flex flex-col gap-2 p-3 flex-1">
        <h4 className="text-[13px] font-semibold text-gray-900 line-clamp-2 leading-snug">
          {title ?? 'タイトル未設定'}
        </h4>

        {tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {tags.slice(0, 3).map((tag) => (
              <span key={tag.id} className="px-1.5 py-0.5 bg-gray-100 rounded-full text-[10px] text-gray-500 border border-gray-200">
                #{tag.name}
              </span>
            ))}
          </div>
        )}

        {aiReason && (
          <p className="text-[11px] text-gray-500 leading-snug line-clamp-2">{aiReason}</p>
        )}

        <div className="mt-auto flex items-center gap-2 pt-1">
          <button
            type="button"
            onClick={onLike}
            disabled={isLiking}
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] transition flex-shrink-0 ${
              isLiked
                ? 'border-rose-300 bg-rose-50 text-rose-600'
                : 'border-gray-200 bg-white text-gray-700 hover:border-rose-200 hover:text-rose-600'
            } ${isLiking ? 'opacity-70 cursor-not-allowed' : ''}`}
          >
            <Heart size={11} fill={isLiked ? '#f43f5e' : 'none'} className={isLiked ? 'text-rose-500' : ''} />
            {isLiked ? '済み' : '気になる'}
          </button>
          {href !== '#' && (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-[11px] text-gray-500 hover:text-gray-800 ml-auto"
            >
              <Play size={12} />
              作品ページ
            </a>
          )}
        </div>
      </div>
    </article>
  );
}

function sourceLabel(source: string) {
  if (source.includes('personal') || source.includes('user')) return 'あなた向け';
  if (source.includes('trend')) return 'トレンド';
  if (source.includes('fresh') || source.includes('new')) return '新着';
  if (source.includes('prompt') || source.includes('keyword')) return 'キーワード';
  return 'おすすめ';
}

// ── フィルターコンボボックス ──────────────────────────────
function FilterCombobox({
  label,
  color,
  searchTerm,
  onSearchChange,
  options,
  selected,
  onToggle,
  loading: lookupLoading,
}: {
  label: string;
  color: 'rose' | 'indigo';
  searchTerm: string;
  onSearchChange: (v: string) => void;
  options: TagOrPerformer[];
  selected: TagOrPerformer[];
  onToggle: (item: TagOrPerformer) => void;
  loading: boolean;
}) {
  const [open, setOpen] = useState(false);
  const blurTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const accent = color === 'rose'
    ? { ring: 'focus:ring-rose-300', option: 'bg-rose-50 text-rose-600', selected: 'text-[10px] text-rose-500' }
    : { ring: 'focus:ring-indigo-300', option: 'bg-indigo-50 text-indigo-600', selected: 'text-[10px] text-indigo-500' };

  return (
    <Combobox<TagOrPerformer | null> value={null} onChange={(v) => { if (v) onToggle(v); }}>
      <div className="relative">
        <div className={`flex items-center gap-2 rounded-xl border border-gray-200 bg-white/90 px-3 py-2 shadow-sm ${accent.ring}`}>
          <Search size={14} className="text-gray-400 flex-shrink-0" />
          <Combobox.Input
            className="w-full text-sm text-gray-900 placeholder-gray-400 focus:outline-none bg-transparent"
            placeholder={label}
            onChange={(e) => onSearchChange(e.target.value)}
            onFocus={() => { if (blurTimer.current) clearTimeout(blurTimer.current); setOpen(true); }}
            onBlur={() => { blurTimer.current = setTimeout(() => setOpen(false), 150); }}
            displayValue={() => ''}
          />
        </div>
        <Combobox.Options
          static
          className={`absolute z-[60] mt-1 max-h-48 w-full overflow-auto rounded-xl border border-gray-200 bg-white text-sm shadow-lg transition-opacity ${
            open ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
          }`}
        >
          {lookupLoading ? (
            <div className="px-3 py-2 text-xs text-gray-400">検索中…</div>
          ) : options.length === 0 ? (
            <div className="px-3 py-2 text-xs text-gray-400">候補が見つかりません</div>
          ) : (
            options.map((item) => (
              <Combobox.Option
                key={item.id}
                value={item}
                className={({ active }) =>
                  `flex items-center justify-between px-3 py-1.5 text-sm cursor-pointer ${active ? accent.option : 'text-gray-700'}`
                }
              >
                <span>{color === 'rose' ? `#${item.name}` : item.name}</span>
                {selected.some((s) => s.id === item.id) && (
                  <span className={accent.selected}>選択中</span>
                )}
              </Combobox.Option>
            ))
          )}
        </Combobox.Options>
      </div>
    </Combobox>
  );
}

// ── メインページ ─────────────────────────────────────────
export default function FindPage() {
  const [authStatus, setAuthStatus] = useState<'checking' | 'authenticated' | 'guest'>('checking');
  const [keyword, setKeyword] = useState('');
  const [debouncedKeyword, setDebouncedKeyword] = useState('');
  const [selectedTags, setSelectedTags] = useState<TagOrPerformer[]>([]);
  const [appliedTagIds, setAppliedTagIds] = useState<string[]>([]);
  const [tagSearchTerm, setTagSearchTerm] = useState('');
  const [tagSearchResults, setTagSearchResults] = useState<TagOrPerformer[]>([]);
  const [tagLookupLoading, setTagLookupLoading] = useState(false);
  const [likingIds, setLikingIds] = useState<Set<string>>(new Set());
  const [likedIds, setLikedIds] = useState<Set<string>>(new Set());
  const [infoMessage, setInfoMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [filterOpen, setFilterOpen] = useState(false);
  const infoTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const keywordDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const filterDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 認証確認
  useEffect(() => {
    let isMounted = true;
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (isMounted) setAuthStatus(session?.user ? 'authenticated' : 'guest');
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_e, session) => {
      if (isMounted) setAuthStatus(session?.user ? 'authenticated' : 'guest');
    });
    return () => { isMounted = false; listener.subscription.unsubscribe(); };
  }, []);

  const isAuthenticated = authStatus === 'authenticated';
  const isGuest = authStatus === 'guest';

  // キーワードデバウンス（400ms）
  useEffect(() => {
    if (keywordDebounceRef.current) clearTimeout(keywordDebounceRef.current);
    keywordDebounceRef.current = setTimeout(() => setDebouncedKeyword(keyword), 400);
    return () => { if (keywordDebounceRef.current) clearTimeout(keywordDebounceRef.current); };
  }, [keyword]);

  // フィルターデバウンス（600ms）
  useEffect(() => {
    if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current);
    filterDebounceRef.current = setTimeout(() => {
      setAppliedTagIds(selectedTags.map((t) => t.id));
    }, 600);
    return () => { if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current); };
  }, [selectedTags]);

  // タグ検索
  useEffect(() => {
    const term = tagSearchTerm.trim();
    if (term.length < 2) { setTagSearchResults([]); setTagLookupLoading(false); return; }
    let cancelled = false;
    setTagLookupLoading(true);
    const timer = setTimeout(async () => {
      const { data } = await supabase.from('tags').select('id,name').ilike('name', `%${term}%`).order('name').limit(8);
      if (!cancelled) {
        setTagSearchResults((data ?? []).map((r) => ({ id: String(r.id), name: (r as { name?: string }).name ?? '' })));
        setTagLookupLoading(false);
      }
    }, 350);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [tagSearchTerm]);

  useEffect(() => () => { if (infoTimeoutRef.current) clearTimeout(infoTimeoutRef.current); }, []);

  // ユーザーの好みデータ（タグサジェスト用）
  const { data: analysisData } = useAnalysisResults({
    windowDays: 90, includeNope: false, tagLimit: 20, performerLimit: 0, recentLimit: 0,
    enabled: isAuthenticated,
  });
  const topTags = useMemo(() => (analysisData?.top_tags ?? []).slice(0, 10).map((t) => ({ id: t.tag_id, name: t.tag_name })), [analysisData]);

  const displayedTagOptions = tagSearchTerm.trim().length >= 2 ? tagSearchResults : topTags;

  // AIレコメンド（キーワードなし時のメインコンテンツ）
  const aiEnabled = authStatus !== 'checking';
  const { data: aiData, loading: aiLoading } = useAiRecommend({
    limitPerSection: isAuthenticated ? 12 : 8,
    tagIds: isAuthenticated ? appliedTagIds : undefined,
    enabled: aiEnabled,
  });

  // キーワード検索
  const { results: searchResults, loading: searchLoading } = useVideoSearch({
    keyword: debouncedKeyword,
    tagIds: appliedTagIds.length > 0 ? appliedTagIds : undefined,
  });

  const isSearchMode = debouncedKeyword.trim().length > 0;

  // AI結果をフラットなリストに
  const aiSections = useMemo(() => {
    const sections = aiData?.sections ?? [];
    return isGuest ? sections.filter((s) => s.id.toLowerCase().includes('trend') || s.id.toLowerCase().includes('fresh')) : sections;
  }, [aiData, isGuest]);

  const aiItems = useMemo(() => {
    return aiSections.flatMap((section) =>
      section.items.map((item) => ({ ...item, sectionId: section.id }))
    );
  }, [aiSections]);

  const showInfoMessage = useCallback((type: 'success' | 'error', text: string) => {
    if (infoTimeoutRef.current) clearTimeout(infoTimeoutRef.current);
    setInfoMessage({ type, text });
    infoTimeoutRef.current = setTimeout(() => setInfoMessage(null), 3200);
  }, []);

  const handleLike = useCallback(async (videoId: string, sectionId?: string, source?: string, score?: number | null, reason?: string) => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      try { window.dispatchEvent(new Event('open-register-modal')); } catch { /* noop */ }
      showInfoMessage('error', '「気になる」を保存するにはログインしてください。');
      return;
    }
    setLikingIds((prev) => new Set(prev).add(videoId));
    try {
      const { error } = await supabase.from('user_book_decisions').upsert({
        user_id: user.id,
        book_id: videoId,
        decision_type: 'like',
        recommendation_source: source ?? sectionId ?? 'find_page',
        recommendation_score: typeof score === 'number' ? score : null,
        recommendation_model_version: null,
        recommendation_params: reason ? { recommendation_reason_summary: reason } : null,
        recommendation_type: sectionId ? 'like_on_ai_search' : 'like_on_search',
      }, { onConflict: 'user_id,book_id' });
      if (error) {
        showInfoMessage('error', '保存に失敗しました。');
      } else {
        setLikedIds((prev) => new Set(prev).add(videoId));
        showInfoMessage('success', '「気になる」に追加しました。');
      }
    } finally {
      setLikingIds((prev) => { const next = new Set(prev); next.delete(videoId); return next; });
    }
  }, [showInfoMessage]);

  const toggleTag = (item: TagOrPerformer) => {
    setSelectedTags((prev) =>
      prev.some((t) => t.id === item.id) ? prev.filter((t) => t.id !== item.id) : prev.length >= 5 ? prev : [...prev, item]
    );
  };
  const clearFilters = () => {
    setSelectedTags([]);
    setTagSearchTerm('');
  };

  const hasFilters = selectedTags.length > 0;

  if (authStatus === 'checking') {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-white/60" />
      </main>
    );
  }

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-6 relative">
      {infoMessage && (
        <div
          className={`fixed bottom-4 left-1/2 -translate-x-1/2 z-[200] px-4 py-3 rounded-full shadow-lg border text-sm whitespace-nowrap ${
            infoMessage.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
              : 'bg-rose-50 border-rose-200 text-rose-800'
          }`}
          role="status"
          aria-live="polite"
        >
          {infoMessage.text}
        </div>
      )}

      <div className="max-w-7xl mx-auto flex flex-col gap-5">
        {/* ── ヘッダー + 検索バー ── */}
        <div className="px-4 sm:px-0 flex flex-col gap-3">
          <div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight">さがす</h1>
            <p className="text-sm text-white/60 mt-0.5">キーワード・タグから漫画を見つける</p>
          </div>

          {/* 検索バー */}
          <div className="relative">
            <div className="flex items-center gap-3 rounded-2xl bg-white/95 border border-white/40 shadow-lg px-4 py-3">
              {searchLoading ? (
                <Loader2 size={18} className="text-gray-400 animate-spin flex-shrink-0" />
              ) : (
                <Search size={18} className="text-gray-400 flex-shrink-0" />
              )}
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="タイトルやキーワードで検索…"
                className="flex-1 text-base text-gray-900 placeholder-gray-400 focus:outline-none bg-transparent"
              />
              {keyword && (
                <button type="button" onClick={() => setKeyword('')} className="text-gray-400 hover:text-gray-600">
                  <X size={16} />
                </button>
              )}
            </div>
          </div>

          {/* フィルター折りたたみ */}
          <div className="rounded-2xl bg-white/10 border border-white/20 overflow-hidden">
            <button
              type="button"
              onClick={() => setFilterOpen((v) => !v)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-white/80 hover:text-white transition"
            >
              <span className="flex items-center gap-2">
                タグでしぼる
                {hasFilters && (
                  <span className="bg-rose-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                    {selectedTags.length}
                  </span>
                )}
              </span>
              {filterOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {filterOpen && (
              <div className="px-4 pb-4 flex flex-col gap-3">
                {/* 選択済みチップ */}
                {hasFilters && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {selectedTags.map((tag) => (
                      <button
                        key={tag.id}
                        type="button"
                        onClick={() => toggleTag(tag)}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-500 text-white border border-rose-400"
                      >
                        #{tag.name}
                        <X size={11} />
                      </button>
                    ))}
                    <button
                      type="button"
                      onClick={clearFilters}
                      className="px-2.5 py-1 rounded-full text-xs text-white/60 hover:text-white border border-white/20 hover:border-white/40 transition"
                    >
                      クリア
                    </button>
                  </div>
                )}

                <div className="relative z-[80]">
                  <div className="space-y-1.5">
                    <p className="text-xs text-white/50 font-medium">タグ（最大5件）</p>
                    <FilterCombobox
                      label="タグ名を入力（例: 制服）"
                      color="rose"
                      searchTerm={tagSearchTerm}
                      onSearchChange={setTagSearchTerm}
                      options={displayedTagOptions}
                      selected={selectedTags}
                      onToggle={toggleTag}
                      loading={tagLookupLoading}
                    />
                    {/* トップタグのクイック選択 */}
                    {isAuthenticated && topTags.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {topTags.slice(0, 6).map((tag) => {
                          const active = selectedTags.some((t) => t.id === tag.id);
                          return (
                            <button
                              key={tag.id}
                              type="button"
                              onClick={() => toggleTag(tag)}
                              className={`px-2 py-0.5 rounded-full text-[11px] font-medium border transition ${
                                active ? 'bg-rose-500 text-white border-rose-500' : 'bg-white/10 text-white/70 border-white/20 hover:border-rose-300 hover:text-rose-300'
                              }`}
                            >
                              #{tag.name}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>

                {!isAuthenticated && (
                  <p className="text-xs text-white/40 text-center">
                    ログインするとあなたの好み履歴からタグを素早く選べます
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── コンテンツエリア ── */}
        {isSearchMode ? (
          // キーワード検索モード
          <SearchResultsSection
            keyword={debouncedKeyword}
            results={searchResults}
            loading={searchLoading}
            likedIds={likedIds}
            likingIds={likingIds}
            onLike={handleLike}
          />
        ) : (
          // AIレコメンドモード
          <AiPicksSection
            items={aiItems}
            loading={aiLoading}
            isGuest={isGuest}
            likedIds={likedIds}
            likingIds={likingIds}
            onLike={handleLike}
          />
        )}
      </div>
    </main>
  );
}

// ── 検索結果セクション ─────────────────────────────────
function SearchResultsSection({
  keyword,
  results,
  loading,
  likedIds,
  likingIds,
  onLike,
}: {
  keyword: string;
  results: VideoSearchItem[];
  loading: boolean;
  likedIds: Set<string>;
  likingIds: Set<string>;
  onLike: (id: string) => void;
}) {
  return (
    <div className="px-4 sm:px-0 flex flex-col gap-4">
      <div className="flex items-baseline gap-2">
        <h2 className="text-base font-bold text-white">
          「{keyword}」の検索結果
        </h2>
        {!loading && (
          <span className="text-sm text-white/50">{results.length}件</span>
        )}
      </div>

      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="rounded-xl bg-white/10 aspect-[3/4] animate-pulse" />
          ))}
        </div>
      ) : results.length === 0 ? (
        <div className="rounded-2xl bg-white/10 border border-white/20 p-8 text-center text-white/60">
          <p className="text-base font-medium">見つかりませんでした</p>
          <p className="text-sm mt-1">別のキーワードやタグを試してみてください</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {results.map((video) => (
            <VideoCard
              key={video.id}
              id={video.id}
              title={video.title}
              thumbnail_url={video.thumbnail_url}
              product_url={video.product_url}
              product_released_at={video.product_released_at}
              tags={video.tags}
              isLiked={likedIds.has(video.id)}
              isLiking={likingIds.has(video.id)}
              onLike={() => onLike(video.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── AIレコメンドセクション ────────────────────────────
function AiPicksSection({
  items,
  loading,
  isGuest,
  likedIds,
  likingIds,
  onLike,
}: {
  items: Array<ReturnType<typeof flattenAiItems>[number]>;
  loading: boolean;
  isGuest: boolean;
  likedIds: Set<string>;
  likingIds: Set<string>;
  onLike: (id: string, sectionId: string, source: string, score?: number | null, reason?: string) => void;
}) {
  return (
    <div className="px-4 sm:px-0 flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Sparkles size={16} className="text-rose-400" />
        <h2 className="text-base font-bold text-white">
          {isGuest ? 'トレンド・新着' : 'あなたへのおすすめ'}
        </h2>
        {isGuest && (
          <span className="text-xs text-white/40 ml-auto">ログインでパーソナライズされます</span>
        )}
      </div>

      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="rounded-xl bg-white/10 aspect-[3/4] animate-pulse" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="rounded-2xl bg-white/10 border border-white/20 p-8 text-center text-white/60">
          <p className="text-sm">おすすめが見つかりませんでした</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {items.map((item) => (
            <VideoCard
              key={`${item.sectionId}:${item.id}`}
              id={item.id}
              title={item.title}
              thumbnail_url={item.thumbnail_url}
              product_url={item.product_url ?? null}
              product_released_at={item.metrics.product_released_at ?? null}
              tags={item.tags}
              aiReason={item.reason.summary}
              aiSource={item.metrics.source}
              isLiked={likedIds.has(item.id)}
              isLiking={likingIds.has(item.id)}
              onLike={() => onLike(item.id, item.sectionId, item.metrics.source, item.metrics.score, item.reason.summary)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// TypeScript helper for the flatten type
function flattenAiItems(sections: AiRecommendSection[]) {
  return sections.flatMap((s) => s.items.map((item) => ({ ...item, sectionId: s.id })));
}

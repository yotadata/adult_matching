'use client';

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { X, Play, Heart, Eye, ChevronDown, ChevronUp, Brain, Hand, Bot, Target, LockOpen, ExternalLink, type LucideIcon } from 'lucide-react';
import { trackEvent } from '@/lib/analytics';
import OnboardingModal from '@/components/OnboardingModal';

type VideoItem = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  sample_video_url: string | null;
  embed_url: string | null;
  product_url: string | null;
  product_released_at: string | null;
  performers: { id: string; name: string }[];
  tags: { id: string; name: string }[];
  source: string;
  score: number | null;
};

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const FANZA_AFFILIATE_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';
const EMBED_USER_INTERVAL = 10; // N回いいねするたびにembed-userを呼ぶ

function toFanzaEmbedUrl(externalId: string | null): string {
  if (!externalId) return '';
  return `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${FANZA_AFFILIATE_ID}/cid=${externalId}/size=1280_720/`;
}

const SOURCE_BADGE_COLORS: Record<string, string> = {
  exploitation: 'bg-violet-600 text-white',
  exploitation_tag: 'bg-orange-500 text-white',
  popularity: 'bg-blue-600 text-white',
  exploration: 'bg-green-700 text-white',
};

const SOURCE_BORDER_COLORS: Record<string, string> = {
  exploitation: 'border-violet-500',
  exploitation_tag: 'border-orange-400',
  popularity: 'border-blue-500',
  exploration: 'border-green-600',
};

function GridPage() {
  const searchParams = useSearchParams();
  const isDebug = searchParams.get('debug') === '1';
  const [bannerOpen, setBannerOpen] = useState(() => {
    if (typeof window === 'undefined') return true;
    return localStorage.getItem('grid_banner_closed') !== '1';
  });
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [selected, setSelected] = useState<VideoItem | null>(null);
  const [showVideo, setShowVideo] = useState(false);
  const [likedIds, setLikedIds] = useState<Set<string>>(new Set());
  const [viewedIds, setViewedIds] = useState<Set<string>>(new Set());
  const [loadedIds, setLoadedIds] = useState<Set<string>>(new Set());
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [showLoginNudge, setShowLoginNudge] = useState(false);
  const [nopedIds, setNopedIds] = useState<Set<string>>(new Set());
  const [showOnboarding, setShowOnboarding] = useState(false);
  const preferredTagIds = useRef<string[]>([]);
  const guestLikeCountRef = useRef(0);
  const guestLikeCountTotalRef = useRef(0);
  const overlayHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const OVERLAY_HIDE_DELAY_MS = 700;
  const loadedVideoIds = useRef<Set<string>>(new Set());
  const likeCountSinceEmbed = useRef(0);

  const fetchVideos = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': session?.access_token
          ? `Bearer ${session.access_token}`
          : `Bearer ${SUPABASE_ANON_KEY}`,
      };
      const res = await fetch(`${SUPABASE_URL}/functions/v1/videos-grid`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          limit: 30,
          exclude_ids: Array.from(loadedVideoIds.current),
          preferred_tag_ids: preferredTagIds.current.length > 0 ? preferredTagIds.current : undefined,
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (isDebug && data._debug) console.log('[grid debug]', JSON.stringify(data._debug));
      const newVideos: VideoItem[] = (data.videos ?? []).filter(
        (v: VideoItem) => !loadedVideoIds.current.has(v.id)
      );
      if (newVideos.length === 0) {
        setHasMore(false);
        return;
      }
      newVideos.forEach((v) => loadedVideoIds.current.add(v.id));
      setVideos((prev) => [...prev, ...newVideos]);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [loading, hasMore]);

  // 認証状態を取得
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsLoggedIn(Boolean(session?.user));
      trackEvent('recommend_session_start', { source: 'grid' });
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_e, session) => {
      setIsLoggedIn(Boolean(session?.user));
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  // オンボーディング表示チェック
  useEffect(() => {
    const done = localStorage.getItem('onboarding_done');
    if (!done) {
      setShowOnboarding(true);
    }
  }, []);

  // DBからいいね済みIDを取得
  useEffect(() => {
    (async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;
      const { data } = await supabase
        .from('user_video_decisions')
        .select('video_id')
        .eq('user_id', user.id)
        .in('decision_type', ['swipe_like', 'grid_like']);
      if (data && data.length > 0) {
        setLikedIds(new Set(data.map((d) => d.video_id)));
      }
    })();
  }, []);

  // 初回ロード（オンボーディング中でも背景で読み込む）
  useEffect(() => {
    const saved = localStorage.getItem('onboarding_tags');
    if (saved) {
      try { preferredTagIds.current = JSON.parse(saved); } catch { /* ignore */ }
    }
    fetchVideos();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 無限スクロール
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) fetchVideos();
      },
      { threshold: 0.1 }
    );
    if (loaderRef.current) observer.observe(loaderRef.current);
    return () => observer.disconnect();
  }, [fetchVideos]);

  const toAffiliateUrl = (raw?: string | null) => {
    const AF_ID = 'yotadata2-001';
    if (!raw) return '';
    if (raw.startsWith('https://al.fanza.co.jp/')) {
      try {
        const u = new URL(raw);
        u.searchParams.set('af_id', AF_ID);
        return u.toString();
      } catch {}
    }
    return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
  };

  const handleLike = useCallback(async (video: VideoItem) => {
    if (likedIds.has(video.id)) return;
    setLikedIds((prev) => new Set([...prev, video.id]));
    window.dispatchEvent(new Event('like-added'));

    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      guestLikeCountRef.current += 1;
      guestLikeCountTotalRef.current += 1;
      trackEvent('guest_swipe', { swipe_count: guestLikeCountTotalRef.current, decision_type: 'grid_like' });
      if (guestLikeCountRef.current === 3 || guestLikeCountRef.current % 10 === 0) {
        setShowLoginNudge(true);
        trackEvent('login_nudge_shown', { like_count: guestLikeCountRef.current });
      }
      return;
    }

    const { error } = await supabase.from('user_video_decisions').insert({
      user_id: user.id,
      video_id: video.id,
      decision_type: 'grid_like',
      recommendation_source: video.source ?? null,
    });
    if (error) {
      console.error('Error inserting like:', error.message);
    }
    likeCountSinceEmbed.current += 1;
    if (likeCountSinceEmbed.current >= EMBED_USER_INTERVAL) {
      likeCountSinceEmbed.current = 0;
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        supabase.functions.invoke('embed-user', {
          headers: { Authorization: `Bearer ${session.access_token}` },
        }).catch(() => {});
      }
    }
  }, [likedIds]);

  const handleNope = useCallback(async (video: VideoItem) => {
    if (nopedIds.has(video.id)) return;
    setNopedIds((prev) => new Set([...prev, video.id]));
    setSelected(null);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;
    await supabase.from('user_video_decisions').insert({
      user_id: user.id,
      video_id: video.id,
      decision_type: 'grid_nope',
    }).then(({ error }) => {
      if (error) console.error('Error inserting nope:', error.message);
    });
  }, [nopedIds]);

  const handleOnboardingComplete = (tagIds: string[]) => {
    preferredTagIds.current = tagIds;
    // タグ付きで最初からフェッチし直す
    setVideos([]);
    loadedVideoIds.current = new Set();
    setHasMore(true);
    setShowOnboarding(false);
    fetchVideos();
  };

  return (
    <div className="min-h-screen bg-[#0d1117]" style={{ paddingTop: '52px' }}>
      {/* オンボーディングモーダル */}
      {showOnboarding && (
        <OnboardingModal onComplete={handleOnboardingComplete} />
      )}
      {/* ゲスト向けログイン nudge トースト */}
      {showLoginNudge && isLoggedIn === false && (
        <>
          {/* モバイル: 小さいトースト */}
          <div className="sm:hidden fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-[#1c1f26] border border-violet-500/50 rounded-2xl px-4 py-3 shadow-2xl shadow-violet-900/30 max-w-[320px] w-[90vw]">
            <Heart size={18} className="text-pink-400 flex-shrink-0" fill="currentColor" />
            <div className="flex-1 min-w-0">
              <p className="text-white text-[12px] font-bold leading-snug">いいね履歴を保存しませんか？</p>
              <p className="text-[#8b949e] text-[10px] mt-0.5">ログインするとAIがあなたの好みを学習します</p>
            </div>
            <div className="flex flex-col gap-1 flex-shrink-0">
              <button
                onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_clicked', { like_count: guestLikeCountRef.current }); window.dispatchEvent(new Event('open-auth-modal')); }}
                className="px-3 py-1 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-[11px] font-bold transition-colors"
              >
                登録
              </button>
              <button
                onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_dismissed', { like_count: guestLikeCountRef.current }); }}
                className="px-3 py-1 rounded-lg text-[#8b949e] hover:text-white text-[11px] transition-colors text-center"
              >
                後で
              </button>
            </div>
          </div>
          {/* PC: 画面下部の幅広バナー */}
          <div className="hidden sm:flex fixed bottom-0 left-0 right-0 z-50 items-center justify-between gap-6 bg-gradient-to-r from-violet-950/95 via-[#1c1f26]/98 to-violet-950/95 border-t border-violet-500/40 px-8 py-4 shadow-2xl shadow-violet-900/40 backdrop-blur-sm">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-pink-500/20 border border-pink-500/40 flex items-center justify-center flex-shrink-0">
                <Heart size={20} className="text-pink-400" fill="currentColor" />
              </div>
              <div>
                <p className="text-white text-sm font-bold">いいね履歴を保存しませんか？</p>
                <p className="text-[#8b949e] text-xs mt-0.5">ログインするとAIがあなたの性癖を学習し、より精度の高いおすすめが届きます</p>
              </div>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <button
                onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_dismissed', { like_count: guestLikeCountRef.current }); }}
                className="px-4 py-2 rounded-lg text-[#8b949e] hover:text-white text-sm transition-colors"
              >
                後で
              </button>
              <button
                onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_clicked', { like_count: guestLikeCountRef.current }); window.dispatchEvent(new Event('open-auth-modal')); }}
                className="px-6 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold transition-colors shadow-lg shadow-violet-900/50"
              >
                無料で登録
              </button>
            </div>
          </div>
        </>
      )}
      {/* デバッグ: 左上オーバーレイ（セッション集計） */}
      {isDebug && videos.length > 0 && (() => {
        const counts: Record<string, number> = {};
        const scores: Record<string, number[]> = {};
        for (const v of videos) {
          counts[v.source] = (counts[v.source] ?? 0) + 1;
          if (v.score != null) {
            if (!scores[v.source]) scores[v.source] = [];
            scores[v.source].push(v.score);
          }
        }
        const total = videos.length;
        const COLORS: Record<string, string> = {
          exploitation: 'text-violet-400',
          popularity: 'text-blue-400',
          exploration: 'text-green-400',
        };
        return (
          <div className="fixed top-[60px] left-2 z-50 bg-black/80 backdrop-blur-sm border border-white/10 rounded-xl px-3 py-2 text-[10px] font-mono flex flex-col gap-1 min-w-[160px]">
            <span className="text-[#8b949e]">total:{total}</span>
            {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([src, cnt]) => {
              const avg = scores[src]?.length
                ? (scores[src].reduce((a, b) => a + b, 0) / scores[src].length).toFixed(3)
                : '–';
              const max = scores[src]?.length ? Math.max(...scores[src]).toFixed(3) : '–';
              return (
                <div key={src} className="flex flex-col gap-0">
                  <span className={`font-bold ${COLORS[src] ?? 'text-white'}`}>{src} {cnt} ({((cnt / total) * 100).toFixed(0)}%)</span>
                  <span className="text-yellow-300 pl-1">avg:{avg} max:{max}</span>
                </div>
              );
            })}
          </div>
        );
      })()}
      {/* 説明バナー */}
      <div className="max-w-4xl mx-auto px-3 pt-4">
      <div className="mt-3 mb-5 rounded-xl border border-violet-500/40 bg-gradient-to-br from-violet-950/60 to-[#161b22] overflow-hidden shadow-lg shadow-violet-900/20">
        <button
          className="w-full flex items-center justify-between px-4 py-3 text-left"
          onClick={() => {
            const next = !bannerOpen;
            setBannerOpen(next);
            localStorage.setItem('grid_banner_closed', next ? '0' : '1');
          }}
        >
          <span className="flex items-center gap-2 text-[#e6edf3] text-base font-extrabold">
            <Brain size={16} className="text-violet-400" />
            性癖ラボとは？
          </span>
          {bannerOpen ? <ChevronUp size={16} className="text-[#8b949e]" /> : <ChevronDown size={16} className="text-[#8b949e]" />}
        </button>
        {bannerOpen && (
          <div className="px-4 pb-4 space-y-3">
            <p className="text-[#8b949e] text-xs leading-relaxed">
              AIがあなたの「いいね」を学習し、好みにぴったりのAV動画を提案するサービスです。<br />
              いいねすればするほど推薦精度が上がり、自分だけのおすすめフィードが育っていきます。
            </p>
            <div className="grid grid-cols-3 gap-2">
              {([
                { Icon: Hand,   title: 'いいねする', desc: '気になった動画にいいね',    color: 'text-pink-400' },
                { Icon: Bot,    title: 'AIが学習',   desc: 'あなたの性癖パターンを解析', color: 'text-blue-400' },
                { Icon: Target, title: '精度UP',     desc: 'いいねが増えるほど精確に',   color: 'text-green-400' },
              ] as { Icon: LucideIcon; title: string; desc: string; color: string }[]).map((item) => (
                <div key={item.title} className="bg-[#0d1117] rounded-lg p-2.5 text-center">
                  <item.Icon size={20} className={`mx-auto mb-1 ${item.color}`} strokeWidth={1.5} />
                  <div className="text-[#e6edf3] text-[11px] font-bold">{item.title}</div>
                  <div className="text-[#8b949e] text-[10px] mt-0.5">{item.desc}</div>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-[#8b949e] bg-[#0d1117] rounded-lg px-3 py-2">
              <LockOpen size={13} className="flex-shrink-0 text-[#8b949e]" />
              <span>メールアドレス不要・無料で今すぐ使えます。登録するといいね履歴が永続保存されます。</span>
            </div>
            {isLoggedIn === false && (
              <button
                onClick={() => window.dispatchEvent(new Event('open-auth-modal'))}
                className="w-full py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold transition-colors"
              >
                ログイン / 新規登録
              </button>
            )}
          </div>
        )}
      </div>
      </div>{/* /max-w-4xl */}

      {/* グリッド: モバイル=2カラム縦長、PC=Pinterestスタイル */}
      <div className="px-2 sm:px-3 py-4">
      {/* モバイル: 2カラム masonry（columns） */}
      <div className="columns-2 gap-2 space-y-2 sm:hidden">
        {videos.map((video) => (
          <div
            key={video.id}
            className={`break-inside-avoid rounded-xl overflow-hidden cursor-pointer bg-[#161b22] border transition-all shadow-md ${
              loadedIds.has(video.id) ? 'opacity-100' : 'opacity-0'
            } ${
              likedIds.has(video.id)
                ? 'border-pink-500/60'
                : isDebug
                  ? (SOURCE_BORDER_COLORS[video.source] ?? 'border-[#30363d]')
                  : 'border-[#30363d] hover:border-violet-500/60'
            }`}
            onClick={() => {
              if (overlayHideTimer.current) clearTimeout(overlayHideTimer.current);
              setShowVideo(false);
              setSelected(video);
              setViewedIds((prev) => new Set([...prev, video.id]));
            }}
          >
            {/* サムネイル */}
            <div className="w-full bg-black relative aspect-[7/10]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={video.thumbnail_url ?? ''}
                alt={video.title ?? ''}
                className="w-full h-full object-cover object-right"
                loading="lazy"
                onLoad={() => setLoadedIds((prev) => new Set([...prev, video.id]))}
              />
              {viewedIds.has(video.id) && !likedIds.has(video.id) && <div className="absolute inset-0 bg-black/60 pointer-events-none" />}
              {likedIds.has(video.id) && <div className="absolute inset-0 bg-pink-500/30 pointer-events-none" />}
              {nopedIds.has(video.id) && <div className="absolute inset-0 bg-black/70 pointer-events-none" />}
              <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/80 to-transparent pointer-events-none" />
              {/* 再生ボタン（中央・常時表示） */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-10 h-10 rounded-full bg-black/50 border border-white/30 flex items-center justify-center">
                  <Play size={18} className="text-white ml-0.5" fill="currentColor" />
                </div>
              </div>
              {/* ハートボタン（右下） */}
              {nopedIds.has(video.id) ? (
                <div className="absolute bottom-2 right-2 w-8 h-8 rounded-full bg-white/20 flex items-center justify-center shadow-md pointer-events-none">
                  <X size={14} className="text-white/80" strokeWidth={2.5} />
                </div>
              ) : (
                <button
                  className={`absolute bottom-2 right-2 w-8 h-8 rounded-full flex items-center justify-center transition-colors shadow-lg ${likedIds.has(video.id) ? 'bg-pink-500' : 'bg-black/50 hover:bg-pink-500/80'}`}
                  onClick={(e) => { e.stopPropagation(); handleLike(video); }}
                  aria-label="いいね"
                >
                  <Heart size={15} className="text-white" fill={likedIds.has(video.id) ? 'white' : 'none'} strokeWidth={2} />
                </button>
              )}
              {isDebug && (
                <div className="absolute bottom-1 left-1">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${SOURCE_BADGE_COLORS[video.source] ?? 'bg-gray-600 text-white'}`}>{video.source}</span>
                </div>
              )}
            </div>
            {/* タイトル・タグ */}
            <div className="px-2 pt-1.5 pb-2">
              <p className="text-[11px] font-semibold text-[#e6edf3] leading-snug mb-1">
                {video.title}
              </p>
              {(video.tags as { id: string; name: string }[])?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {(video.tags as { id: string; name: string }[]).slice(0, 2).map((t) => (
                    <span key={t.id} className="text-[9px] bg-violet-900/40 border border-violet-700/40 text-violet-300 rounded-full px-1.5 py-0.5">
                      {t.name}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      {/* PC: Pinterestスタイル */}
      <div className="hidden sm:block columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-2 space-y-2">
        {videos.map((video) => (
          <div
            key={video.id}
            className={`break-inside-avoid rounded-2xl overflow-hidden cursor-pointer bg-[#161b22] border transition-all shadow-md ${
              loadedIds.has(video.id) ? 'opacity-100' : 'opacity-0'
            } ${
              likedIds.has(video.id)
                ? 'border-pink-500/60'
                : isDebug
                  ? (SOURCE_BORDER_COLORS[video.source] ?? 'border-[#30363d]')
                  : 'border-[#30363d] hover:border-violet-500/60'
            }`}
            onClick={() => {
              if (overlayHideTimer.current) clearTimeout(overlayHideTimer.current);
              setShowVideo(false);
              setSelected(video);
              setViewedIds((prev) => new Set([...prev, video.id]));
            }}
          >
            {/* サムネイル */}
            <div className="w-full bg-black relative group">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={(video.thumbnail_vertical_url?.replace('ps.jpg', 'pl.jpg')) || video.thumbnail_url || ''}
                alt={video.title ?? ''}
                className="w-full object-cover"
                loading="lazy"
                onLoad={() => setLoadedIds((prev) => new Set([...prev, video.id]))}
              />
              {/* 既読オーバーレイ（いいね済みでない場合のみ） */}
              {viewedIds.has(video.id) && !likedIds.has(video.id) && (
                <div className="absolute inset-0 bg-black/60 pointer-events-none" />
              )}
              {/* いいね済みオーバーレイ */}
              {likedIds.has(video.id) && (
                <div className="absolute inset-0 bg-pink-500/40 pointer-events-none" />
              )}
              {/* 興味なしオーバーレイ */}
              {nopedIds.has(video.id) && (
                <div className="absolute inset-0 bg-black/70 pointer-events-none" />
              )}
              {/* 常時表示: 下部グラデーション + アクションヒント */}
              <div className="absolute inset-x-0 bottom-0 h-14 bg-gradient-to-t from-black/70 to-transparent pointer-events-none" />
              {/* 再生ボタン（モバイル: 常時表示、PC: hover時のみ） */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-9 h-9 rounded-full bg-black/30 flex items-center justify-center sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                  <Play size={16} className="text-white/80 ml-0.5" fill="currentColor" />
                </div>
              </div>
              {/* いいね/興味なしボタン（常時・右下） */}
              {nopedIds.has(video.id) ? (
                <div className="absolute bottom-2 right-2 w-8 h-8 rounded-full bg-white/20 flex items-center justify-center shadow-md pointer-events-none">
                  <X size={14} className="text-white/80" strokeWidth={2.5} />
                </div>
              ) : (
                <button
                  className={`absolute bottom-2 right-2 w-8 h-8 rounded-full flex items-center justify-center transition-colors shadow-lg ${
                    likedIds.has(video.id)
                      ? 'bg-pink-500'
                      : 'bg-black/50 hover:bg-pink-500/80'
                  }`}
                  onClick={(e) => { e.stopPropagation(); handleLike(video); }}
                  aria-label="いいね"
                >
                  <Heart size={15} className="text-white" fill={likedIds.has(video.id) ? 'white' : 'none'} strokeWidth={2} />
                </button>
              )}
              {/* 既読バッジ（いいね済みでない場合のみ・左上） */}
              {viewedIds.has(video.id) && !likedIds.has(video.id) && (
                <div className="absolute top-1.5 left-1.5 w-5 h-5 rounded-full bg-black/60 border border-white/20 flex items-center justify-center pointer-events-none">
                  <Eye size={10} className="text-white/70" />
                </div>
              )}
              {/* デバッグバッジ */}
              {isDebug && (
                <div className="absolute bottom-1 left-1 flex flex-col gap-0.5 items-start">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${SOURCE_BADGE_COLORS[video.source] ?? 'bg-gray-600 text-white'}`}>
                    {video.source}
                  </span>
                  {video.score != null && (
                    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-black/70 text-yellow-300">
                      {video.score.toFixed(3)}
                    </span>
                  )}
                </div>
              )}
            </div>
            {/* テキスト情報 */}
            <div className="px-2.5 py-2">
              <p className="text-[11px] font-bold text-[#e6edf3] line-clamp-2 leading-snug">{video.title}</p>
              {(video.performers as { id: string; name: string }[])?.length > 0 && (
                <p className="text-[10px] text-[#8b949e] mt-0.5 line-clamp-1">
                  {(video.performers as { id: string; name: string }[]).map((p) => p.name).join(' / ')}
                </p>
              )}
              {(video.tags as { id: string; name: string }[])?.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {(video.tags as { id: string; name: string }[]).slice(0, 3).map((t) => (
                    <span key={t.id} className="text-[9px] bg-violet-900/40 border border-violet-700/40 text-violet-300 rounded-full px-1.5 py-0.5">
                      {t.name}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      </div>{/* PC columns / モバイルgrid wrapper */}

      {/* ローダー */}
      <div ref={loaderRef} className="h-16 flex items-center justify-center">
        {loading && <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />}
        {!hasMore && <p className="text-[#8b949e] text-xs">すべて表示しました</p>}
      </div>

      {/* 動画モーダル */}
      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="relative w-full max-w-2xl bg-white rounded-2xl border border-gray-200 shadow-xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* 閉じるボタン */}
            <button
              onClick={() => setSelected(null)}
              className="absolute top-2 right-2 z-20 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80"
            >
              <X size={16} />
            </button>

            {/* 動画エリア（SwipeCardと同じ4:3） */}
            <div className="relative w-full aspect-[4/3] bg-black/90 flex items-center justify-center rounded-t-2xl overflow-hidden">
              {showVideo === false && (
                <div
                  className="absolute inset-0 w-full h-full bg-contain bg-no-repeat bg-center flex items-center justify-center z-10 cursor-pointer"
                  style={{
                    backgroundImage: selected.thumbnail_url ? `url(${selected.thumbnail_url})` : undefined,
                    backgroundColor: selected.thumbnail_url ? undefined : '#1f2937',
                  }}
                  onClick={() => setShowVideo(true)}
                >
                  <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                    <Play className="text-white w-16 h-16 opacity-80" fill="white" />
                  </div>
                  <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-white/80 bg-black/40 px-2 py-0.5 rounded whitespace-nowrap">
                    注: 再生には最大2回のクリックが必要な場合があります
                  </div>
                </div>
              )}
              <iframe
                scrolling="no"
                referrerPolicy="no-referrer"
                src={toFanzaEmbedUrl(selected.external_id)}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
                loading="eager"
                onLoad={() => {
                  if (overlayHideTimer.current) clearTimeout(overlayHideTimer.current);
                  overlayHideTimer.current = setTimeout(() => {
                    setShowVideo(true);
                    overlayHideTimer.current = null;
                  }, OVERLAY_HIDE_DELAY_MS);
                }}
                className="absolute top-0 left-0 w-full h-full overflow-hidden"
              />
            </div>

            {/* 情報エリア */}
            <div className="flex flex-col text-gray-800 px-4 py-3 gap-2">
              <h2 className="text-base font-extrabold tracking-tight line-clamp-2">{selected.title}</h2>
              {selected.performers && (selected.performers as { id: string; name: string }[]).length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {(selected.performers as { id: string; name: string }[]).map((p) => (
                    <span key={p.id} className="rounded-full bg-pink-500/70 px-2 py-1 text-[11px] font-bold text-white">
                      {p.name}
                    </span>
                  ))}
                </div>
              )}
              {selected.tags && (selected.tags as { id: string; name: string }[]).length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {(selected.tags as { id: string; name: string }[]).map((t) => (
                    <span key={t.id} className="rounded-full bg-purple-600/60 px-2 py-1 text-[11px] font-bold text-white">
                      {t.name}
                    </span>
                  ))}
                </div>
              )}
              {/* アクションボタン */}
              <div className="flex flex-col gap-2 mt-1">
                <div className="flex gap-2">
                  <button
                    onClick={() => handleNope(selected)}
                    disabled={nopedIds.has(selected.id)}
                    className="flex items-center gap-1.5 justify-center flex-1 py-2.5 rounded-lg text-sm font-bold transition-colors border border-gray-300 bg-white text-gray-400 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <X size={15} />
                    {nopedIds.has(selected.id) ? '除外済み' : '興味なし'}
                  </button>
                  <button
                    onClick={() => handleLike(selected)}
                    className={`flex items-center gap-1.5 justify-center flex-1 py-2.5 rounded-lg text-sm font-bold transition-colors border ${
                      likedIds.has(selected.id)
                        ? 'bg-pink-500 border-pink-500 text-white'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-pink-50 hover:border-pink-300'
                    }`}
                  >
                    <Heart size={15} fill={likedIds.has(selected.id) ? 'white' : 'none'} />
                    {likedIds.has(selected.id) ? 'いいね済み' : 'いいね'}
                  </button>
                </div>
                {likedIds.has(selected.id) && selected.product_url && (
                  <a
                    href={toAffiliateUrl(selected.product_url)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 justify-center w-full py-2 rounded-lg bg-[#f0f0f0] hover:bg-[#e0e0e0] text-gray-500 text-xs font-medium transition-colors"
                  >
                    <ExternalLink size={12} />
                    本編を見る
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function GridPageWrapper() {
  return (
    <Suspense>
      <GridPage />
    </Suspense>
  );
}

'use client';

import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import { useState, useRef, useEffect, useCallback, Suspense } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";
import useMediaQuery from "@/hooks/useMediaQuery";
import useWindowSize from "@/hooks/useWindowSize";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { supabase } from "@/lib/supabase";
import { trackEvent, generateSessionId } from "@/lib/analytics";
import { resolveEmbedUrl } from "@/lib/videoMeta";
import { ChevronsLeft, Heart, List } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { useDecisionCount } from "@/hooks/useDecisionCount";

interface VideoFromApi {
  id: number;
  title: string;
  description: string;
  external_id: string;
  thumbnail_url: string;
  sample_video_url?: string;
  preview_video_url?: string;
  product_url?: string;
  product_released_at?: string;
  performers: { id: string; name: string }[];
  tags: { id: string; name: string }[];
  source?: string | null;
  video_source?: string | null;
  score?: number | null;
  model_version?: string | null;
  params?: Record<string, unknown> | null;
  image_urls?: string[] | null;
}

interface GuestDecision {
  video_id: CardData['id'];
  decision_type: 'swipe_like' | 'swipe_nope';
  created_at: string;
  recommendation_source?: string | null;
  recommendation_score?: number | null;
  recommendation_model_version?: string | null;
  recommendation_params?: Record<string, unknown> | null;
}

interface VideosFeedMetadata {
  swipes_until_next_embed: number;
  decision_count: number;
}

const ORIGINAL_GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
const LEFT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
const RIGHT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
function SwipePage() {

  const searchParams = useSearchParams();
  const isDebug = searchParams.get('debug') === '1';
  const [cards, setCards] = useState<CardData[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isFetchingVideos, setIsFetchingVideos] = useState(false);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const { height: windowHeight } = useWindowSize();
  const [cardWidth, setCardWidth] = useState<number | undefined>(400);
  const [swipesUntilNextEmbed, setSwipesUntilNextEmbed] = useState<number | null>(null);
  const { incrementDecisionCount } = useDecisionCount();
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [authReady, setAuthReady] = useState<boolean>(false);
  const [showLoginNudge, setShowLoginNudge] = useState(false);
  const guestLikeCountRef = useRef(0);
  const guestSwipeCountRef = useRef(0);
  const mainRef = useRef<HTMLDivElement | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const sessionStartRef = useRef<number | null>(null);
  const sessionCompletedRef = useRef<boolean>(false);
  const samplePlayedRef = useRef<boolean>(false);
  const samplePlayTimestampRef = useRef<number | null>(null);
  const samplePlayCountRef = useRef<number>(0);
  const previousCardRef = useRef<CardData | null>(null);
  const currentPositionRef = useRef<number>(0);
  const likeButtonRef = useRef<HTMLButtonElement | null>(null);
  const skipButtonRef = useRef<HTMLButtonElement | null>(null);
  const likedListButtonRef = useRef<HTMLButtonElement | null>(null);

  const toIsoString = useCallback((ms: number | null) => (ms != null ? new Date(ms).toISOString() : undefined), []);

  const videoAspectRatio = 4 / 3;
  const activeCard = activeIndex < cards.length ? cards[activeIndex] : null;

  const startInteractionSession = useCallback((card: CardData, position: number) => {
    const sessionId = generateSessionId();
    sessionIdRef.current = sessionId;
    sessionStartRef.current = Date.now();
    sessionCompletedRef.current = false;
    samplePlayedRef.current = false;
    samplePlayTimestampRef.current = null;
    samplePlayCountRef.current = 0;
    currentPositionRef.current = position;
    const source = card.recommendationSource ?? 'videos_feed';
    trackEvent('recommend_session_start', {
      session_id: sessionId,
      video_id: card.id,
      position,
      has_session: isLoggedIn,
      source,
      session_started_at: toIsoString(sessionStartRef.current),
      recommendation_score: typeof card.recommendationScore === 'number' ? card.recommendationScore : undefined,
      recommendation_model_version: card.recommendationModelVersion ?? undefined,
    });
  }, [isLoggedIn, toIsoString]);

  const abandonInteractionSession = useCallback((card: CardData) => {
    if (!sessionIdRef.current || sessionStartRef.current === null || sessionCompletedRef.current) {
      return;
    }
    const source = card.recommendationSource ?? 'videos_feed';
    trackEvent('recommend_session_abandon', {
      session_id: sessionIdRef.current,
      video_id: card.id,
      position: currentPositionRef.current,
      has_session: isLoggedIn,
      source,
      session_started_at: toIsoString(sessionStartRef.current),
      session_abandoned_at: toIsoString(Date.now()),
      sample_played: samplePlayedRef.current ? 1 : 0,
      sample_play_count: samplePlayCountRef.current,
    });
    sessionCompletedRef.current = true;
    sessionIdRef.current = null;
    sessionStartRef.current = null;
    samplePlayedRef.current = false;
    samplePlayTimestampRef.current = null;
    samplePlayCountRef.current = 0;
  }, [isLoggedIn, toIsoString]);

  const emitDecisionEvents = useCallback((card: CardData, decisionType: 'swipe_like' | 'swipe_nope') => {
    if (!card) return;
    if (!sessionIdRef.current || sessionStartRef.current === null) {
      startInteractionSession(card, currentPositionRef.current);
    }
    if (!sessionIdRef.current || sessionStartRef.current === null) return;
    if (sessionCompletedRef.current) return;
    const now = Date.now();
    const source = card.recommendationSource ?? 'videos_feed';
    const baseParams = {
      session_id: sessionIdRef.current,
      video_id: card.id,
      position: currentPositionRef.current,
      has_session: isLoggedIn,
      source,
      decision_type: decisionType,
    } as const;
    trackEvent('recommend_decision', {
      ...baseParams,
      session_started_at: toIsoString(sessionStartRef.current),
      decision_at: toIsoString(now),
      sample_played: samplePlayedRef.current ? 1 : 0,
      sample_last_play_at: toIsoString(samplePlayTimestampRef.current),
      sample_play_count: samplePlayCountRef.current,
    });
    trackEvent('recommend_session_complete', {
      ...baseParams,
      session_started_at: toIsoString(sessionStartRef.current),
      session_completed_at: toIsoString(now),
      sample_played: samplePlayedRef.current ? 1 : 0,
      sample_last_play_at: toIsoString(samplePlayTimestampRef.current),
      sample_play_count: samplePlayCountRef.current,
    });
    sessionCompletedRef.current = true;
    sessionIdRef.current = null;
    sessionStartRef.current = null;
    samplePlayedRef.current = false;
    samplePlayTimestampRef.current = null;
    samplePlayCountRef.current = 0;
    previousCardRef.current = null;
  }, [isLoggedIn, startInteractionSession, toIsoString]);

  const handleSamplePlay = useCallback((card: CardData, surface: 'desktop' | 'mobile') => {
    if (!card) return;
    if (sessionCompletedRef.current) return;
    if (!sessionIdRef.current || sessionStartRef.current === null) return;
    if (previousCardRef.current && previousCardRef.current.id !== card.id) return;
    const now = Date.now();
    samplePlayedRef.current = true;
    samplePlayCountRef.current += 1;
    samplePlayTimestampRef.current = now;
    const source = card.recommendationSource ?? 'videos_feed';
    trackEvent('recommend_sample_play', {
      session_id: sessionIdRef.current,
      video_id: card.id,
      position: currentPositionRef.current,
      has_session: isLoggedIn,
      source,
      sample_type: card.sampleVideoUrl ? 'sample' : 'embed',
      session_started_at: toIsoString(sessionStartRef.current),
      sample_play_at: toIsoString(now),
      play_count_in_session: samplePlayCountRef.current,
      surface,
    });
  }, [isLoggedIn, toIsoString]);
 
  const refetchVideos = useCallback(async () => {
    const requestStartedAt = Date.now();
    try {
      setIsFetchingVideos(true);

      const { data: { session } } = await supabase.auth.getSession();
      const headers: HeadersInit = {};
      if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
      const timeoutMs = 20000;
      const timeoutPromise = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('videos-feed timeout')), timeoutMs));
      const savedTags = localStorage.getItem('onboarding_tags');
      const preferredTagIds = savedTags ? (() => { try { return JSON.parse(savedTags); } catch { return []; } })() : [];
      const invokePromise = supabase.functions.invoke('videos-feed', {
        headers,
        body: preferredTagIds.length > 0 ? { preferred_tag_ids: preferredTagIds } : {},
      });
      const { data, error } = await Promise.race([invokePromise, timeoutPromise]);
      if (error) {
        console.error('Error from videos-feed invoke:', error);
        trackEvent('recommend_fetch', {
          status: 'error',
          source: 'videos_feed',
          response_ms: Date.now() - requestStartedAt,
          has_session: isLoggedIn,
          error_message: error.message,
        });
        return;
      }

      const responseData = data as { videos: VideoFromApi[], metadata: VideosFeedMetadata };
      const fetchedVideos = responseData.videos || [];
      const metadata = responseData.metadata;

      if (metadata) {
        setSwipesUntilNextEmbed(metadata.swipes_until_next_embed);
      }

      const normalizeHttps = (u?: string) => u?.startsWith('http://') ? u.replace('http://', 'https://') : u;
      const fetchedCards: CardData[] = fetchedVideos.map((video) => {
        const normalizedSampleUrl = normalizeHttps(video.sample_video_url);
        const normalizedPreviewUrl = normalizeHttps(video.preview_video_url);
        const embed = resolveEmbedUrl({
          source: video.video_source,
          externalId: video.external_id,
          sampleVideoUrl: normalizedSampleUrl || normalizedPreviewUrl,
        });
        return {
          id: video.id,
          title: video.title,
          genre: video.tags.map((tag) => tag.name),
          description: video.description,
          videoUrl: embed?.url ?? '',
          sampleVideoUrl: normalizedSampleUrl || normalizedPreviewUrl,
          embedUrl: embed?.url ?? '',
          embedType: embed?.type ?? 'iframe',
          thumbnail_url: video.thumbnail_url,
          product_released_at: video.product_released_at,
          performers: video.performers,
          tags: video.tags,
          productUrl: normalizeHttps(video.product_url) || undefined,
          recommendationSource: video.source ?? null,
          recommendationScore: typeof video.score === 'number' ? video.score : null,
          recommendationModelVersion: video.model_version ?? null,
          recommendationParams: video.params ?? null,
          source: video.video_source ?? null,
          image_urls: video.image_urls ?? null,
        };
      });
      setCards(fetchedCards);
      setActiveIndex(0);

      trackEvent('recommend_fetch', {
        status: 'success',
        source: 'videos_feed',
        response_ms: Date.now() - requestStartedAt,
        has_session: isLoggedIn,
        videos_count: fetchedCards.length,
        swipes_until_next_embed: metadata?.swipes_until_next_embed,
        decision_count: metadata?.decision_count,
      });
    } catch (err) {
      console.error('UNCAUGHT ERROR:', err);
      const message = err instanceof Error ? err.message : String(err);
      trackEvent('recommend_fetch', {
        status: 'error',
        source: 'videos_feed',
        response_ms: Date.now() - requestStartedAt,
        has_session: isLoggedIn,
        error_message: message,
      });
    } finally {
      setIsFetchingVideos(false);
    }
  }, [isLoggedIn]);

  useEffect(() => {
    const recalc = () => {
      if (!isMobile) {
        const mainH = mainRef.current?.clientHeight;
        if (mainH && mainH > 0) {
          const targetVideoHeight = mainH * (3 / 5);
          const calculatedCardWidth = targetVideoHeight * videoAspectRatio;
          setCardWidth(calculatedCardWidth);
          return;
        }
      }
      if (windowHeight) {
        const targetVideoHeight = (!isMobile ? windowHeight * (3 / 5) : windowHeight / 2);
        const calculatedCardWidth = targetVideoHeight * videoAspectRatio;
        setCardWidth(calculatedCardWidth);
      }
    };
    recalc();
  }, [windowHeight, videoAspectRatio, isMobile]);

  useEffect(() => {
    if (!authReady) return;
    const currentCard = activeCard;
    const previousCard = previousCardRef.current;

    if (previousCard && (!currentCard || previousCard.id !== currentCard.id)) {
      abandonInteractionSession(previousCard);
    }

    if (currentCard && (!previousCard || previousCard.id !== currentCard.id)) {
      startInteractionSession(currentCard, activeIndex);
    }

    previousCardRef.current = currentCard;
  }, [activeCard, activeIndex, authReady, abandonInteractionSession, startInteractionSession]);

  useEffect(() => {
    return () => {
      if (previousCardRef.current) {
        abandonInteractionSession(previousCardRef.current);
      }
    };
  }, [abandonInteractionSession]);
  
  const getGuestDecisions = useCallback((): GuestDecision[] => {
    try {
      const raw = typeof window !== 'undefined' ? localStorage.getItem('guest_decisions_v1') : null;
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  }, []);

  const setGuestDecisions = useCallback((arr: GuestDecision[]) => {
    localStorage.setItem('guest_decisions_v1', JSON.stringify(arr));
  }, []);

  const flushGuestDecisions = useCallback(async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;
    const items = getGuestDecisions();
    if (!items.length) return;
    const batchSize = 100;
    const VALID_TYPES = new Set(['swipe_like', 'swipe_nope', 'grid_like', 'grid_nope']);
    const validItems = items.filter((d) => VALID_TYPES.has(d.decision_type));
    // 同一video_idが複数ある場合は最後のものを優先して重複除去
    const deduped = [...validItems.reduce((map, d) => {
      map.set(d.video_id, d);
      return map;
    }, new Map()).values()];
    for (let i = 0; i < deduped.length; i += batchSize) {
      const chunk = deduped.slice(i, i + batchSize).map((d) => ({
        user_id: user.id,
        video_id: d.video_id,
        decision_type: d.decision_type,
        created_at: d.created_at ?? new Date().toISOString(),
        recommendation_source: d.recommendation_source ?? null,
        recommendation_score: d.recommendation_score ?? null,
        recommendation_model_version: d.recommendation_model_version ?? null,
        recommendation_params: d.recommendation_params ?? null,
      }));
      const { error } = await supabase
        .from('user_video_decisions')
        .upsert(chunk, { onConflict: 'user_id,video_id' });
      if (error) {
        console.error('Error flushing guest decisions:', error?.message ?? error);
        return;
      }
    }
    setGuestDecisions([]);
  }, [getGuestDecisions, setGuestDecisions]);

  useEffect(() => {
    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(!!session?.user);
      setAuthReady(true);
      if (!!session?.user) {
        flushGuestDecisions();
      }
    });

    return () => {
      authListener.subscription.unsubscribe();
    };
  }, [flushGuestDecisions]);

  useEffect(() => {
    if (!authReady) {
      return;
    }
    refetchVideos();
  }, [authReady, refetchVideos]);


  const handleSwipe = async (direction: 'left' | 'right') => {
    const card = activeCard;
    if (!card) return;
    const decisionType = direction === 'right' ? 'swipe_like' : 'swipe_nope';

    if (isLoggedIn) {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { error } = await supabase.from('user_video_decisions').insert({
          user_id: user.id,
          video_id: card.id,
          decision_type: decisionType,
          recommendation_source: card.recommendationSource ?? null,
          recommendation_score: card.recommendationScore ?? null,
          recommendation_model_version: card.recommendationModelVersion ?? null,
          recommendation_params: card.recommendationParams ?? null,
        });
        if (error) {
          console.error(`Error inserting ${decisionType} decision:`, error);
        }
      }
    } else {
      const current = getGuestDecisions();
      current.push({
        video_id: card.id,
        decision_type: decisionType,
        created_at: new Date().toISOString(),
        recommendation_source: card.recommendationSource ?? null,
        recommendation_score: card.recommendationScore ?? null,
        recommendation_model_version: card.recommendationModelVersion ?? null,
        recommendation_params: card.recommendationParams ?? null,
      });
      setGuestDecisions(current);
      guestSwipeCountRef.current += 1;
      trackEvent('guest_swipe', { swipe_count: guestSwipeCountRef.current, decision_type: decisionType });
      if (decisionType === 'swipe_like') {
        guestLikeCountRef.current += 1;
        if (guestLikeCountRef.current === 3 || guestLikeCountRef.current % 10 === 0) {
          setShowLoginNudge(true);
          trackEvent('login_nudge_shown', { swipe_count: guestSwipeCountRef.current, like_count: guestLikeCountRef.current });
        }
      }
    }

    if (decisionType === 'swipe_like') {
      window.dispatchEvent(new Event('like-added'));
    }
    emitDecisionEvents(card, decisionType);
    trackEvent('swipe_action', {
      direction,
      decision_type: decisionType,
      video_id: card.id,
      logged_in: isLoggedIn ? 1 : 0,
      recommendation_source: card.recommendationSource ?? 'videos_feed',
    });
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
    incrementDecisionCount();

    // Decrement swipe counter and trigger embed-user if it reaches zero
    if (isLoggedIn && swipesUntilNextEmbed !== null) {
      const newCount = swipesUntilNextEmbed - 1;
      setSwipesUntilNextEmbed(newCount);

      if (newCount <= 0) {
        console.log("Triggering embed-user API call...");
        const { data: { session } } = await supabase.auth.getSession();
        if (session) {
          supabase.functions.invoke('embed-user', {
            headers: { Authorization: `Bearer ${session.access_token}` }
          }).then(({ error }) => {
            if (error) {
              console.error("Error calling embed-user:", error.message);
            } else {
              console.log("embed-user API call successful.");
              // Optionally, refetch videos to get the new countdown
              refetchVideos();
            }
          });
        }
      }
    }
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const handleDrag = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (info.offset.x > 50) {
      setCurrentGradient(RIGHT_SWIPE_GRADIENT);
    } else if (info.offset.x < -50) {
      setCurrentGradient(LEFT_SWIPE_GRADIENT);
    } else {
      setCurrentGradient(ORIGINAL_GRADIENT);
    }
  };

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (Math.abs(info.offset.x) <= 100) {
      setCurrentGradient(ORIGINAL_GRADIENT);
    }
  };

  useEffect(() => {
    if (cards.length > 0 && activeIndex >= cards.length && !isFetchingVideos) {
      refetchVideos();
    }
  }, [activeIndex, cards.length, isFetchingVideos, refetchVideos]);

  return (
    <motion.div
      className="flex flex-col items-center h-screen overflow-hidden pt-[51px]"
      style={{ background: currentGradient }}
      transition={{ duration: 0.3 }}
    >
      {/* デバッグ: 左上オーバーレイ（セッション集計） */}
      {isDebug && cards.length > 0 && (() => {
        const relevant = cards.slice(activeIndex);
        const all = cards;
        const counts: Record<string, number> = {};
        const scores: Record<string, number[]> = {};
        for (const c of all) {
          const src = c.recommendationSource ?? 'unknown';
          counts[src] = (counts[src] ?? 0) + 1;
          if (c.recommendationScore != null) {
            if (!scores[src]) scores[src] = [];
            scores[src].push(c.recommendationScore);
          }
        }
        const total = all.length;
        const COLORS: Record<string, string> = {
          exploitation: 'text-violet-400',
          popularity: 'text-blue-400',
          exploration: 'text-green-400',
        };
        return (
          <div className="fixed top-[60px] left-2 z-50 bg-black/80 backdrop-blur-sm border border-white/10 rounded-xl px-3 py-2 text-[10px] font-mono flex flex-col gap-1 min-w-[160px]">
            <span className="text-[#8b949e]">loaded:{total} rem:{relevant.length}</span>
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
      {/* デバッグ: 右上オーバーレイ（現在のカード情報） */}
      {isDebug && activeCard && (
        <div className="fixed top-[60px] right-2 z-50 bg-black/80 backdrop-blur-sm border border-white/10 rounded-xl px-3 py-2 text-[10px] font-mono flex flex-col gap-0.5 min-w-[140px] items-end">
          <span className="text-white font-bold">{activeCard.recommendationSource ?? 'unknown'}</span>
          {typeof activeCard.recommendationScore === 'number' && (
            <span className="text-yellow-300">score: {activeCard.recommendationScore.toFixed(4)}</span>
          )}
          {activeCard.recommendationModelVersion && (
            <span className="text-cyan-300">{activeCard.recommendationModelVersion}</span>
          )}
        </div>
      )}
      {/* ゲスト向けログイン nudge トースト */}
      {showLoginNudge && !isLoggedIn && (
        <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-[#1c1f26] border border-violet-500/50 rounded-2xl px-4 py-3 shadow-2xl shadow-violet-900/30 max-w-[320px] w-[90vw]">
          <Heart size={18} className="text-pink-400 flex-shrink-0" fill="currentColor" />
          <div className="flex-1 min-w-0">
            <p className="text-white text-[12px] font-bold leading-snug">いいね履歴を保存しませんか？</p>
            <p className="text-[#8b949e] text-[10px] mt-0.5">ログインするとAIがあなたの好みを学習します</p>
          </div>
          <div className="flex flex-col gap-1 flex-shrink-0">
            <button
              onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_clicked', { swipe_count: guestSwipeCountRef.current }); window.dispatchEvent(new Event('open-auth-modal')); }}
              className="px-3 py-1 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-[11px] font-bold transition-colors"
            >
              登録
            </button>
            <button
              onClick={() => { setShowLoginNudge(false); trackEvent('login_nudge_dismissed', { swipe_count: guestSwipeCountRef.current }); }}
              className="px-3 py-1 rounded-lg text-[#8b949e] hover:text-white text-[11px] transition-colors text-center"
            >
              後で
            </button>
          </div>
        </div>
      )}
      <main
        ref={mainRef}
        className={`flex-grow flex w-full relative ${isMobile ? 'flex-col h-full' : 'items-center justify-center pt-10'}`}
        style={isMobile ? { paddingTop: `0px` } : {}}
      >
        <AnimatePresence mode="wait">
          {activeCard ? (
            isMobile ? (
              <MobileVideoLayout
                cardData={activeCard}
                onSkip={() => handleSwipe('left')}
                onLike={() => handleSwipe('right')}
                onSamplePlay={(card) => handleSamplePlay(card, 'mobile')}
                skipButtonRef={skipButtonRef}
                likeButtonRef={likeButtonRef}
                likedListButtonRef={likedListButtonRef}
              />
            ) : (
              <SwipeCard
                ref={cardRef}
                key={activeCard.id}
                cardData={activeCard}
                onSwipe={handleSwipe}
                onDrag={handleDrag}
                onDragEnd={handleDragEnd}
                cardWidth={cardWidth}
                canSwipe={true}
                onSamplePlay={(card) => handleSamplePlay(card, 'desktop')}
              />
            )
          ) : (
            isFetchingVideos ? (
              <div className="flex items-center justify-center w-full h-full">
                <div
                  className="w-12 h-12 rounded-full border-4 border-gray-300 border-t-violet-500 animate-spin"
                  role="status"
                  aria-label="Loading videos"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center w-full h-full text-white/90">
                <p className="mb-3">おすすめ候補は以上です。</p>
                <button
                  onClick={refetchVideos}
                  className="px-4 py-2 rounded-md bg-white/20 hover:bg-white/30 backdrop-blur border border-white/40"
                >
                  おすすめを再取得
                </button>
              </div>
            )
          )}
        </AnimatePresence>
      </main>
      {!isMobile && (
        <footer className="py-8 mx-auto" style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}>
          {activeCard && (
            <>
              <div className="mx-auto w-full flex items-center justify-center gap-6 py-3">
              <button
                onClick={() => triggerSwipe('left')}
                ref={skipButtonRef}
                className="w-20 h-20 rounded-full bg-[#6C757D] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="スキップ"
                title="スキップ"
              >
                <ChevronsLeft size={36} className="text-white" />
              </button>
              <button
                onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
                ref={likedListButtonRef}
                className="w-[60px] h-[60px] rounded-full bg-[#BEBEBE] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="気になるリスト"
                title="気になるリスト"
              >
                <List size={28} className="text-white" />
              </button>
              <button
                onClick={() => triggerSwipe('right')}
                ref={likeButtonRef}
                className="w-20 h-20 rounded-full bg-[#FF6B81] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="気になる"
                title="気になる"
              >
                <Heart size={36} className="text-white" />
              </button>
            </div>
            </>
          )}
        </footer>
      )}

    </motion.div>
  );
}

export default function Home() {
  return (
    <Suspense>
      <SwipePage />
    </Suspense>
  );
}

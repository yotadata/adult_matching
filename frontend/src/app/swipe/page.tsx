'use client';

import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import { Suspense, useState, useRef, useEffect, useCallback, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import useMediaQuery from "@/hooks/useMediaQuery";
import useWindowSize from "@/hooks/useWindowSize";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { supabase } from "@/lib/supabase";
import { trackEvent, generateSessionId } from "@/lib/analytics";
import { ChevronsLeft, Heart, List } from "lucide-react";
import { useDecisionCount } from "@/hooks/useDecisionCount";
import OnboardingSlides from "@/components/OnboardingSlides";
import SpotlightTutorial from "@/components/SpotlightTutorial";
import { useSearchParams } from "next/navigation";

interface VideoFromApi {
  id: number;
  title: string;
  description: string;
  external_id: string;
  thumbnail_url: string;
  thumbnail_vertical_url?: string;
  sample_video_url?: string;
  preview_video_url?: string;
  product_url?: string;
  product_released_at?: string;
  performers: { id: string; name: string }[];
  tags: { id: string; name: string }[];
  source?: string | null;
  score?: number | null;
  model_version?: string | null;
  params?: Record<string, unknown> | null;
}

interface GuestDecision {
  video_id: CardData['id'];
  decision_type: 'like' | 'nope';
  created_at: string;
  recommendation_source?: string | null;
  recommendation_score?: number | null;
  recommendation_model_version?: string | null;
  recommendation_params?: Record<string, unknown> | null;
  recommendation_type?: string | null;
}

interface UserStatsBucket {
  likes: number;
  total: number;
  like_rate: number;
}

interface UserStats {
  window: number;
  like_rate: number;
  by_source: Partial<Record<'exploitation' | 'popularity' | 'exploration', UserStatsBucket>>;
  by_score: Partial<Record<'low' | 'mid' | 'high', UserStatsBucket>>;
}

interface VideosFeedMetadata {
  swipes_until_next_embed: number;
  decision_count: number;
  user_stats: UserStats | null;
}

const ORIGINAL_GRADIENT = 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)';
const LEFT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
const RIGHT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
const ONBOARDING_STORAGE_KEY = 'seihekiLab_hasSeenOnboardingSlides';
const SPOTLIGHT_STORAGE_KEY = 'seihekiLab_hasSeenSpotlightTutorial';

export default function Home() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-r from-[#C4C8E3] via-[#F7D7E0] to-[#F9C9D6]">
        <div className="text-white/70 text-sm">Loading swipe experience…</div>
      </div>
    }>
      <SwipePageContent />
    </Suspense>
  );
}

function SwipePageContent() {

  const [cards, setCards] = useState<CardData[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isFetchingVideos, setIsFetchingVideos] = useState(false);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const { height: windowHeight } = useWindowSize();
  const [cardWidth, setCardWidth] = useState<number | undefined>(400);
  const [swipesUntilNextEmbed, setSwipesUntilNextEmbed] = useState<number | null>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const { decisionCount, incrementDecisionCount } = useDecisionCount();
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const isLoggedInRef = useRef<boolean>(false);
  const [authReady, setAuthReady] = useState<boolean>(false);
  const guestLimit = Number(process.env.NEXT_PUBLIC_GUEST_DECISIONS_LIMIT || 20);
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
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showSpotlight, setShowSpotlight] = useState(false);
  const [spotlightReady, setSpotlightReady] = useState(false);
  const likedListButtonRef = useRef<HTMLButtonElement | null>(null);
  const onboardingStartedRef = useRef(false);
  const spotlightStartedRef = useRef(false);
  const searchParams = useSearchParams();
  const isDebugMode = useMemo(() => {
    const value = searchParams?.get('debug');
    if (!value) return false;
    return !['0', 'false', 'off'].includes(value.toLowerCase());
  }, [searchParams]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seenOnboarding = localStorage.getItem(ONBOARDING_STORAGE_KEY) === 'true';
    const seenSpotlight = localStorage.getItem(SPOTLIGHT_STORAGE_KEY) === 'true';
    if (!seenOnboarding) {
      setShowOnboarding(true);
    } else if (!seenSpotlight) {
      setShowSpotlight(true);
    }
  }, []);

  useEffect(() => {
    if (!showSpotlight) {
      setSpotlightReady(false);
      return;
    }
    if (typeof window === 'undefined') return;
    let rafId: number | null = null;
    const checkReady = () => {
      if (likeButtonRef.current && skipButtonRef.current && likedListButtonRef.current) {
        setSpotlightReady(true);
        return;
      }
      rafId = window.requestAnimationFrame(checkReady);
    };
    checkReady();
    return () => {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, [showSpotlight]);

  useEffect(() => {
    if (showOnboarding) {
      if (!onboardingStartedRef.current) {
        trackEvent('onboarding_slides_start');
        onboardingStartedRef.current = true;
      }
    } else {
      onboardingStartedRef.current = false;
    }
  }, [showOnboarding]);

  useEffect(() => {
    const visible = showSpotlight && spotlightReady && !showOnboarding;
    if (visible) {
      if (!spotlightStartedRef.current) {
        trackEvent('spotlight_tutorial_start');
        spotlightStartedRef.current = true;
      }
    } else if (!showSpotlight) {
      spotlightStartedRef.current = false;
    }
  }, [showSpotlight, spotlightReady, showOnboarding]);

  const handleFinishOnboarding = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(ONBOARDING_STORAGE_KEY, 'true');
    }
    trackEvent('onboarding_slides_complete');
    setShowOnboarding(false);
    setTimeout(() => setShowSpotlight(true), 0);
  }, []);

  const handleFinishSpotlight = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(SPOTLIGHT_STORAGE_KEY, 'true');
    }
    trackEvent('spotlight_tutorial_complete');
    setShowSpotlight(false);
  }, []);

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

  const emitDecisionEvents = useCallback((card: CardData, decisionType: 'like' | 'nope') => {
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
      const timeoutMs = 12000;
      const timeoutPromise = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('videos-feed timeout')), timeoutMs));
      const invokePromise = supabase.functions.invoke('videos-feed', { headers, body: {} });
      const { data, error } = await Promise.race([invokePromise, timeoutPromise]);
      if (error) {
        console.error('Error from videos-feed invoke:', error);
        trackEvent('recommend_fetch', {
          status: 'error',
          source: 'videos_feed',
          response_ms: Date.now() - requestStartedAt,
          has_session: isLoggedInRef.current,
          error_message: error.message,
        });
        return;
      }

      const responseData = data as { videos: VideoFromApi[], metadata: VideosFeedMetadata };
      const fetchedVideos = responseData.videos || [];
      const metadata = responseData.metadata;

      if (metadata) {
        setSwipesUntilNextEmbed(metadata.swipes_until_next_embed);
        setUserStats(metadata.user_stats ?? null);
      }

      const normalizeHttps = (u?: string) => u?.startsWith('http://') ? u.replace('http://', 'https://') : u;
      const fetchedCards: CardData[] = fetchedVideos.map((video) => {
        const fanzaEmbedUrl = `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/`;
        const normalizedSampleUrl = normalizeHttps(video.sample_video_url);
        const normalizedPreviewUrl = normalizeHttps(video.preview_video_url);
        return {
          id: video.id,
          title: video.title,
          genre: video.tags.map((tag) => tag.name),
          description: video.description,
          videoUrl: fanzaEmbedUrl,
          sampleVideoUrl: normalizedSampleUrl || normalizedPreviewUrl,
          embedUrl: fanzaEmbedUrl,
          thumbnail_url: video.thumbnail_url,
          thumbnailVerticalUrl: video.thumbnail_vertical_url,
          product_released_at: video.product_released_at,
          performers: video.performers,
          tags: video.tags,
          productUrl: normalizeHttps(video.product_url) || undefined,
          recommendationSource: video.source ?? null,
          recommendationScore: typeof video.score === 'number' ? video.score : null,
          recommendationModelVersion: video.model_version ?? null,
          recommendationParams: video.params ?? null,
        };
      });
      setCards(fetchedCards);
      setActiveIndex(0);

      trackEvent('recommend_fetch', {
        status: 'success',
        source: 'videos_feed',
        response_ms: Date.now() - requestStartedAt,
        has_session: isLoggedInRef.current,
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
        has_session: isLoggedInRef.current,
        error_message: message,
      });
    } finally {
      setIsFetchingVideos(false);
    }
  }, []);

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
    for (let i = 0; i < items.length; i += batchSize) {
      const chunk = items.slice(i, i + batchSize).map((d) => ({
        user_id: user.id,
        video_id: d.video_id,
        decision_type: d.decision_type,
        created_at: d.created_at ?? new Date().toISOString(),
        recommendation_source: d.recommendation_source ?? null,
        recommendation_score: d.recommendation_score ?? null,
        recommendation_model_version: d.recommendation_model_version ?? null,
        recommendation_params: d.recommendation_params ?? null,
        recommendation_type: d.recommendation_type ?? 'swipe_feed',
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
      isLoggedInRef.current = !!session?.user;
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

  useEffect(() => {
    (async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) await flushGuestDecisions();
    })();
  }, [flushGuestDecisions]);

  const handleSwipe = async (direction: 'left' | 'right') => {
    const card = activeCard;
    if (!card) return;
    const decisionType = direction === 'right' ? 'like' : 'nope';

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
          recommendation_type: 'swipe_feed',
        });
        if (error) {
          console.error(`Error inserting ${decisionType} decision:`, error);
        }
      }
    } else {
      const current = getGuestDecisions();
      if (current.length >= guestLimit) {
        try { window.dispatchEvent(new Event('open-register-modal')); } catch {}
        return;
      }
      current.push({
        video_id: card.id,
        decision_type: decisionType,
        created_at: new Date().toISOString(),
        recommendation_source: card.recommendationSource ?? null,
        recommendation_score: card.recommendationScore ?? null,
        recommendation_model_version: card.recommendationModelVersion ?? null,
        recommendation_params: card.recommendationParams ?? null,
        recommendation_type: 'swipe_feed',
      });
      setGuestDecisions(current);
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
    if (!isLoggedIn) {
      const current = getGuestDecisions();
      if (current.length >= guestLimit) {
        try { window.dispatchEvent(new Event('open-register-modal')); } catch {}
      }
    }

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
            }
          });
        }
      }
    }
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const handleDrag = (dir: 'left' | 'right' | 'reset') => {
    if (dir === 'right') setCurrentGradient(RIGHT_SWIPE_GRADIENT);
    else if (dir === 'left') setCurrentGradient(LEFT_SWIPE_GRADIENT);
    else setCurrentGradient(ORIGINAL_GRADIENT);
  };

  const handleDragEnd = () => {
    setCurrentGradient(ORIGINAL_GRADIENT);
  };

  useEffect(() => {
    if (cards.length > 0 && activeIndex >= cards.length && !isFetchingVideos) {
      refetchVideos();
    }
  }, [activeIndex, cards.length, isFetchingVideos, refetchVideos]);

  return (
    <motion.div
      className="flex flex-col items-center h-screen overflow-hidden"
      style={{ background: currentGradient }}
      transition={{ duration: 0.3 }}
    >
      {isDebugMode && (() => {
        const score = typeof activeCard?.recommendationScore === 'number' ? activeCard.recommendationScore : null;
        const scoreColor = score === null ? 'text-gray-400' : score >= 0.7 ? 'text-green-300' : score >= 0.5 ? 'text-yellow-300' : 'text-red-400';
        const scoreLabel = score === null ? '—' : score >= 0.7 ? '高' : score >= 0.5 ? '中' : '低';
        const params = activeCard?.recommendationParams ?? cards[activeIndex]?.recommendationParams ?? null;
        const exploit = typeof params?.exploitation_returned === 'number' ? params.exploitation_returned : null;
        const pop = typeof params?.popularity_returned === 'number' ? params.popularity_returned : null;
        const explore = typeof params?.exploration_returned === 'number' ? params.exploration_returned : null;
        const deckSources = exploit !== null ? `E${exploit}/P${pop}/X${explore}` : null;
        return (
          <div className="pointer-events-none fixed top-4 right-4 z-50 max-w-[210px] rounded-md border border-white/20 bg-black/75 px-3 py-2 text-[11px] font-mono text-white shadow-lg backdrop-blur">
            <div className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.2em] text-amber-300/80">
              debug mode
            </div>
            <div className="space-y-0.5 leading-tight">
              {activeCard ? (
                <>
                  <p>source: <span className="text-cyan-300">{activeCard.recommendationSource ?? 'videos_feed'}</span></p>
                  <div>
                    <span>score: </span>
                    <span className={scoreColor}>{score !== null ? score.toFixed(4) : '—'}</span>
                    <span className={`ml-1.5 text-[10px] ${scoreColor}`}>[{scoreLabel}]</span>
                    {score !== null && (
                      <div className="mt-0.5 h-1.5 w-full rounded-full bg-white/10">
                        <div
                          className={`h-full rounded-full ${score >= 0.7 ? 'bg-green-400' : score >= 0.5 ? 'bg-yellow-400' : 'bg-red-400'}`}
                          style={{ width: `${Math.min(100, Math.max(0, (score - 0.3) / 0.7 * 100))}%` }}
                        />
                      </div>
                    )}
                  </div>
                  <p>model: <span className="text-cyan-300">{activeCard.recommendationModelVersion ?? '—'}</span></p>
                </>
              ) : (
                <p className="text-red-400">カードなし</p>
              )}
              <div className="mt-1.5 border-t border-white/10 pt-1.5 space-y-0.5">
                <p>deck: <span className="text-green-300">{cards.length - activeIndex}</span> / {cards.length}</p>
                {deckSources && (
                  <p className="text-[10px]">
                    <span className="text-violet-300">個人</span>={exploit} <span className="text-yellow-300">人気</span>={pop} <span className="text-gray-400">探索</span>={explore}
                  </p>
                )}
                <p>decisions: <span className="text-green-300">{decisionCount}</span></p>
                <p>embed in: <span className="text-green-300">{swipesUntilNextEmbed ?? '—'}</span></p>
                <p>auth: <span className={isLoggedIn ? 'text-green-300' : 'text-red-400'}>{isLoggedIn ? 'in' : 'guest'}</span></p>
              </div>
              {userStats && (
                <div className="mt-1.5 border-t border-white/10 pt-1.5 space-y-0.5">
                  <p className="text-[9px] uppercase tracking-widest text-amber-300/60">好み適合度 (直近{userStats.window}件)</p>
                  <p>like率: <span className={userStats.like_rate >= 0.4 ? 'text-green-300' : userStats.like_rate >= 0.25 ? 'text-yellow-300' : 'text-red-400'}>{(userStats.like_rate * 100).toFixed(0)}%</span></p>
                  <div className="text-[10px] space-y-0.5">
                    {(['exploitation', 'popularity', 'exploration'] as const).map(src => {
                      const d = userStats.by_source[src];
                      if (!d) return null;
                      const label = src === 'exploitation' ? '個人' : src === 'popularity' ? '人気' : '探索';
                      const color = src === 'exploitation' ? 'text-violet-300' : src === 'popularity' ? 'text-yellow-300' : 'text-gray-400';
                      return (
                        <p key={src}><span className={color}>{label}</span>: {(d.like_rate * 100).toFixed(0)}% <span className="text-white/40">({d.likes}/{d.total})</span></p>
                      );
                    })}
                  </div>
                  <div className="text-[10px] space-y-0.5 mt-0.5">
                    {(['low', 'mid', 'high'] as const).map(bucket => {
                      const d = userStats.by_score[bucket];
                      if (!d) return null;
                      const label = bucket === 'low' ? '低' : bucket === 'mid' ? '中' : '高';
                      const color = bucket === 'high' ? 'text-green-300' : bucket === 'mid' ? 'text-yellow-300' : 'text-red-400';
                      return (
                        <p key={bucket}>スコア<span className={color}>{label}</span>: {(d.like_rate * 100).toFixed(0)}% <span className="text-white/40">({d.total}件)</span></p>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })()}
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
                canSwipe={isLoggedIn || decisionCount < guestLimit}
                onSamplePlay={(card) => handleSamplePlay(card, 'desktop')}
              />
            )
          ) : (
            isFetchingVideos ? (
              <div className="flex items-center justify-center w-full h-full">
                <div
                  className="w-12 h-12 rounded-full border-4 border-white/10 border-t-violet-400 animate-spin"
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
      <OnboardingSlides open={showOnboarding} onFinish={handleFinishOnboarding} />
      <SpotlightTutorial
        likeButtonRef={likeButtonRef}
        skipButtonRef={skipButtonRef}
        likedListButtonRef={likedListButtonRef}
        visible={showSpotlight && spotlightReady && !showOnboarding}
        onFinish={handleFinishSpotlight}
      />
    </motion.div>
  );
}

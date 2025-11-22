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

interface VideosFeedMetadata {
  swipes_until_next_embed: number;
  decision_count: number;
}

const ORIGINAL_GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
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
  const { decisionCount, incrementDecisionCount } = useDecisionCount();
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
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
      {isDebugMode && (
        <div className="pointer-events-none fixed top-4 right-4 z-50 max-w-xs rounded-md border border-white/20 bg-black/75 px-3 py-2 text-[11px] font-mono text-white shadow-lg backdrop-blur">
          <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.2em] text-amber-300/80">
            debug mode
          </div>
          {activeCard ? (
            <div className="space-y-0.5 leading-tight">
              <p>source: {activeCard.recommendationSource ?? 'videos_feed'}</p>
              <p>
                score: {typeof activeCard.recommendationScore === 'number'
                  ? activeCard.recommendationScore.toFixed(4)
                  : '—'}
              </p>
            </div>
          ) : (
            <p className="leading-tight">カードなし</p>
          )}
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
                canSwipe={isLoggedIn || decisionCount < guestLimit}
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

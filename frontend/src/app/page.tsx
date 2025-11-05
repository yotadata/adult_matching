'use client';

import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import { useState, useRef, useEffect, useCallback } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";
import useMediaQuery from "@/hooks/useMediaQuery";
import useWindowSize from "@/hooks/useWindowSize";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { supabase } from "@/lib/supabase";
import { ThumbsDown, Heart, List } from "lucide-react";
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
}

const ORIGINAL_GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
const LEFT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
const RIGHT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;

export default function Home() {

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
  
    const videoAspectRatio = 4 / 3;
    const activeCard = activeIndex < cards.length ? cards[activeIndex] : null;
  
    const refetchVideos = useCallback(async () => {
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
          return;
        }

        const responseData = data as { videos: VideoFromApi[], metadata: any };
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
      } catch (err) {
        console.error('UNCAUGHT ERROR:', err);
      } finally {
        setIsFetchingVideos(false);
      }
    }, [setIsFetchingVideos, setCards, setActiveIndex]);
  
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
          recommendation_source: d.recommendation_source ?? null,
          recommendation_score: d.recommendation_score ?? null,
          recommendation_model_version: d.recommendation_model_version ?? null,
          recommendation_params: d.recommendation_params ?? null,
        }));
        const { error } = await supabase.from('user_video_decisions').insert(chunk);
        if (error) {
          console.error('Error flushing guest decisions:', error);
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
    if (activeCard) {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const decisionType = direction === 'right' ? 'like' : 'nope';
        const { error } = await supabase.from('user_video_decisions').insert({
          user_id: user.id,
          video_id: activeCard.id,
          decision_type: decisionType,
          recommendation_source: activeCard.recommendationSource ?? null,
          recommendation_score: activeCard.recommendationScore ?? null,
          recommendation_model_version: activeCard.recommendationModelVersion ?? null,
          recommendation_params: activeCard.recommendationParams ?? null,
        });
        if (error) {
          console.error(`Error inserting ${decisionType} decision:`, error);
        }
      } else {
        const current = getGuestDecisions();
        if (current.length >= guestLimit) {
          try { window.dispatchEvent(new Event('open-register-modal')); } catch {}
          return;
        }
        const decisionType = direction === 'right' ? 'like' : 'nope';
        current.push({
          video_id: activeCard.id,
          decision_type: decisionType,
          created_at: new Date().toISOString(),
          recommendation_source: activeCard.recommendationSource ?? null,
          recommendation_score: activeCard.recommendationScore ?? null,
          recommendation_model_version: activeCard.recommendationModelVersion ?? null,
          recommendation_params: activeCard.recommendationParams ?? null,
        });
        setGuestDecisions(current);
      }
    }
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
      className="flex flex-col items-center h-screen overflow-hidden"
      style={{ background: currentGradient }}
      transition={{ duration: 0.3 }}
    >
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
            <div className="mx-auto w-full flex items-center justify-center gap-6 py-3">
              <button
                onClick={() => triggerSwipe('left')}
                className="w-20 h-20 rounded-full bg-[#6C757D] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="イマイチ"
                title="イマイチ"
              >
                <ThumbsDown size={36} className="text-white" />
              </button>
              <button
                onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
                className="w-[60px] h-[60px] rounded-full bg-[#BEBEBE] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="お気に入りリスト"
                title="お気に入りリスト"
              >
                <List size={28} className="text-white" />
              </button>
              <button
                onClick={() => triggerSwipe('right')}
                className="w-20 h-20 rounded-full bg-[#FF6B81] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="好み"
                title="好み"
              >
                <Heart size={36} className="text-white" />
              </button>
            </div>
          )}
        </footer>
      )}
    </motion.div>
  );
}

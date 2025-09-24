'use client';

import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";
// 使い方カードは一旦非表示（読み込みも停止）
import dynamic from "next/dynamic";
import useMediaQuery from "@/hooks/useMediaQuery";
import useWindowSize from "@/hooks/useWindowSize";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { supabase } from "@/lib/supabase"; // supabaseクライアントをインポート;
import { ThumbsDown, Heart, List } from "lucide-react";
// ゲージ表示は当面非表示のため読み込まない

// APIから受け取るvideoオブジェクトの型定義
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
}

// 背景グラデーション: 左から C4C8E3, D7D1E3, F7D7E0, F9C9D6 を等間隔
const ORIGINAL_GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
const LEFT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;
const RIGHT_SWIPE_GRADIENT = ORIGINAL_GRADIENT;

export default function Home() {
  

  const [cards, setCards] = useState<CardData[]>([]); // APIからのデータを保持するstate
  const [activeIndex, setActiveIndex] = useState(0);
  const [isFetchingVideos, setIsFetchingVideos] = useState(false);
  const [noMore, setNoMore] = useState(false);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  // 使い方カードは表示しない
  const [showHowToUse, setShowHowToUse] = useState(false);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const { width: windowWidth, height: windowHeight } = useWindowSize();
  const [cardWidth, setCardWidth] = useState<number | undefined>(400); // cardWidthをstateとして管理し、デフォルト値を400に設定
  const [layoutReady, setLayoutReady] = useState(false); // 初期レイアウト完了フラグ
  const getVideoAspectRatio = () => {
    // 4:3 に統一（デザインに合わせる）
    return 4 / 3;
  };
  const videoAspectRatio = getVideoAspectRatio();
  const initialGuestCount = (() => {
    if (typeof window === 'undefined') return 0;
    try {
      const raw = localStorage.getItem('guest_decisions_v1');
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr.length : 0;
    } catch {
      return 0;
    }
  })();
  const [decisionCount, setDecisionCount] = useState<number>(initialGuestCount);
  const [guestDecisionCount, setGuestDecisionCount] = useState<number>(initialGuestCount);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const guestLimit = Number(process.env.NEXT_PUBLIC_GUEST_DECISIONS_LIMIT || 20);
  const [mounted, setMounted] = useState(false);
  // ゲージのターゲット値は未使用（非表示）
  const mainRef = useRef<HTMLDivElement | null>(null);
  const debugResetGauge = (() => {
    const v = (process.env.NEXT_PUBLIC_DEBUG_RESET_GAUGE || '').toString().toLowerCase();
    return v === '1' || v === 'true' || v === 'yes';
  })();
  

  const activeCard = activeIndex < cards.length ? cards[activeIndex] : null; // activeCard の宣言を移動

  // 再取得（将来は user-embedding 更新後に呼ぶ想定）
  const refetchVideos = async () => {
    try {
      setIsFetchingVideos(true);
      setNoMore(false);
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(!!session?.user);
      const headers: HeadersInit = {};
      if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
      // Debug: log just before invoking the Edge Function to confirm code path
      try { console.log('[Home] invoking videos-feed with auth:', !!session?.access_token); } catch {}
      // Add a timeout guard so the UI never hangs forever if the request is blocked
      const timeoutMs = 12000;
      const timeoutPromise = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('videos-feed timeout')), timeoutMs));
      const invokePromise = supabase.functions.invoke('videos-feed', { headers, body: {} });
      const { data, error } = await Promise.race([invokePromise, timeoutPromise]);
      if (error) {
        console.error('Error refetching videos:', error);
        return;
      }
      const normalizeHttps = (u?: string) => u?.startsWith('http://') ? u.replace('http://', 'https://') : u;
      const fetchedCards: CardData[] = (data as VideoFromApi[]).map((video) => {
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
        };
      });
      setCards(fetchedCards);
      setActiveIndex(0);
      if (fetchedCards.length === 0) setNoMore(true);
    } catch (err) {
      console.error('[Home] refetchVideos failed:', err);
    } finally {
      setIsFetchingVideos(false);
    }
  };
  

  // 初回取得
  useEffect(() => {
    setMounted(true);
    refetchVideos();
  }, []);

  // できるだけ早くゲスト件数を反映（Supabase判定を待たない早期反映）
  useEffect(() => {
    if (debugResetGauge) return;
    try {
      const raw = typeof window !== 'undefined' ? localStorage.getItem('guest_decisions_v1') : null;
      const arr = raw ? JSON.parse(raw) : [];
      if (Array.isArray(arr)) setGuestDecisionCount(arr.length);
      if (!isLoggedIn && Array.isArray(arr)) setDecisionCount(arr.length);
    } catch {}
  }, []);

  useEffect(() => {
    // メイン領域の実寸高さからカード横幅を算出（デスクトップ）
    // 動画はカード高さの 3/5 を占めるため、その高さを基準に幅を決定
    const recalc = () => {
      if (!isMobile) {
        const mainH = mainRef.current?.clientHeight;
        if (mainH && mainH > 0) {
          const targetVideoHeight = mainH * (3 / 5);
          const calculatedCardWidth = targetVideoHeight * videoAspectRatio;
          setCardWidth(calculatedCardWidth);
          setLayoutReady(true);
          return;
        }
      }
      // フォールバック（モバイルや未取得時）
      if (windowHeight) {
        const targetVideoHeight = (!isMobile ? windowHeight * (3 / 5) : windowHeight / 2);
        const calculatedCardWidth = targetVideoHeight * videoAspectRatio;
        setCardWidth(calculatedCardWidth);
        setLayoutReady(true);
      }
    };
    recalc();
  }, [windowHeight, videoAspectRatio, isMobile]);

  // 初期の判断数を読み込む（ログイン済みならDB、未ログインならLocalStorage）
  useEffect(() => {
    const loadDecisionCount = async () => {
      // デバッグ: リロード時にゲージを常に 0 にリセット
      if (debugResetGauge) {
        setDecisionCount(0);
        return;
      }
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        try {
          const raw = localStorage.getItem('guest_decisions_v1');
          const arr = raw ? JSON.parse(raw) : [];
          const n = Array.isArray(arr) ? arr.length : 0;
          setGuestDecisionCount(n);
          setDecisionCount(n);
        } catch {
          setDecisionCount(0);
        }
        return;
      }
      const { count, error } = await supabase
        .from('user_video_decisions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id);
      if (error) {
        console.error('Error fetching decision count:', error);
        return;
      }
      setDecisionCount(count || 0);
    };
    loadDecisionCount();
  }, []);

  // ログイン状態の変化を監視し、ゲスト分をフラッシュ
  useEffect(() => {
    const { data: authListener } = supabase.auth.onAuthStateChange(async (_event, session) => {
      const loggedIn = !!session?.user;
      setIsLoggedIn(loggedIn);
      if (loggedIn) {
        await flushGuestDecisions();
      }
    });
    return () => {
      authListener.subscription.unsubscribe();
    };
  }, []);

  // 初回マウント時、既ログインならローカルの決定をフラッシュ
  useEffect(() => {
    (async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) await flushGuestDecisions();
    })();
  }, []);

  const getGuestDecisions = (): { video_id: number; decision_type: 'like' | 'nope'; created_at: string }[] => {
    try {
      const raw = localStorage.getItem('guest_decisions_v1');
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  };

  const setGuestDecisions = (arr: { video_id: number; decision_type: 'like' | 'nope'; created_at: string }[]) => {
    localStorage.setItem('guest_decisions_v1', JSON.stringify(arr));
    setGuestDecisionCount(arr.length);
  };

  const flushGuestDecisions = async () => {
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
      }));
      const { error } = await supabase.from('user_video_decisions').insert(chunk);
      if (error) {
        console.error('Error flushing guest decisions:', error);
        return;
      }
    }
    setGuestDecisions([]);
    setGuestDecisionCount(0);
  };

  

  const handleSwipe = async (direction: 'left' | 'right') => {
    if (activeCard) { // activeCard が存在する場合のみ処理
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const decisionType = direction === 'right' ? 'like' : 'nope';
        const { error } = await supabase.from('user_video_decisions').insert({
          user_id: user.id,
          video_id: activeCard.id,
          decision_type: decisionType,
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
        current.push({ video_id: activeCard.id, decision_type: decisionType, created_at: new Date().toISOString() });
        setGuestDecisions(current);
      }
    }
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
    setDecisionCount((c) => c + 1);
    if (!isLoggedIn) {
      const current = getGuestDecisions();
      if (current.length >= guestLimit) {
        try { window.dispatchEvent(new Event('open-register-modal')); } catch {}
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

  const handleCloseHowToUse = () => {};

  // 全件スワイプ後は自動で再取得（空が返る場合は手動ボタン表示）
  useEffect(() => {
    if (cards.length > 0 && activeIndex >= cards.length && !isFetchingVideos) {
      refetchVideos();
    }
  }, [activeIndex, cards.length]);

  return (
    <motion.div
      className="flex flex-col items-center h-screen overflow-hidden"
      style={{ background: currentGradient }}
      transition={{ duration: 0.3 }}
    >
      {/* Header はレイアウトで表示 */}
      {/* ゲージは非表示 */}
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
              {/* NOPE (thumb_down, #6C757D) */}
              <button
                onClick={() => triggerSwipe('left')}
                className="w-20 h-20 rounded-full bg-[#6C757D] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="イマイチ"
                title="イマイチ"
              >
                <ThumbsDown size={36} className="text-white" />
              </button>
              {/* Liked list */}
              <button
                onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
                className="w-[60px] h-[60px] rounded-full bg-[#BEBEBE] shadow-2xl drop-shadow-xl active:scale-95 transition flex items-center justify-center leading-none"
                aria-label="お気に入りリスト"
                title="お気に入りリスト"
              >
                <List size={28} className="text-white" />
              </button>
              {/* GOOD (heart, #FF6B81) */}
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
      {/* 使い方カードは非表示 */}
    </motion.div>
  );
}

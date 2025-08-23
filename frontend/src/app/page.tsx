'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";
import HowToUseCard from "@/components/HowToUseCard";
import useMediaQuery from "@/hooks/useMediaQuery";
import useWindowSize from "@/hooks/useWindowSize";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { supabase } from "@/lib/supabase"; // supabaseクライアントをインポート

const ORIGINAL_GRADIENT = 'linear-gradient(to right, #C4C8E3, #D7D1E3, #F7D7E0, #F8DBB9)';
const LEFT_SWIPE_GRADIENT = 'linear-gradient(to right, #AEB4EB, #D7D1E3, #F7D7E0,#F8DBB9)'; // 左端を明るく
const RIGHT_SWIPE_GRADIENT = 'linear-gradient(to right, #C4C8E3,  #D7D1E3, #F7D7E0,#F9CFA0)'; // 右端を明るく

export default function Home() {
  console.log('Home component rendered!');

  const [cards, setCards] = useState<CardData[]>([]); // APIからのデータを保持するstate
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const [showHowToUse, setShowHowToUse] = useState(true);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const { width: windowWidth } = useWindowSize();
  const [cardWidth, setCardWidth] = useState<number | undefined>(undefined); // cardWidthをstateとして管理
  const [headerHeight, setHeaderHeight] = useState(0);

  console.log('isMobile:', isMobile);
  console.log('windowWidth:', windowWidth);
  console.log('cardWidth:', cardWidth);

  // APIから動画データを取得する
  useEffect(() => {
    const fetchVideos = async () => {
      const { data: { session } } = await supabase.auth.getSession();

      const { data, error } = await supabase.functions.invoke('videos-feed', {
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
        },
      });

      if (error) {
        console.error('Error fetching videos:', error);
        return;
      }
      
      // APIレスポンスをCardData形式に変換
      const fetchedCards: CardData[] = data.map((video: any) => {
        const fanzaEmbedUrl = `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/`; // FANZA埋め込みURLを生成
        return {
          id: video.id,
          title: video.title,
          genre: video.genre, // tags から genre に変更
          description: video.description,
          videoUrl: fanzaEmbedUrl, // 生成したURLをセット
        };
      });

      setCards(fetchedCards);
    };

    fetchVideos();
  }, []);

  useEffect(() => {
    const updateCardWidth = () => {
      if (cardRef.current) {
        const swiperCardWidth = cardRef.current.getCardWidth();
        if (swiperCardWidth !== undefined) {
          setCardWidth(swiperCardWidth);
        }
      }
    };

    // 初回レンダリング時とウィンドウサイズ変更時に幅を更新
    updateCardWidth();
    window.addEventListener('resize', updateCardWidth);

    return () => {
      window.removeEventListener('resize', updateCardWidth);
    };
  }, [cardRef.current]); // cardRef.current が変更されたときに実行

  useEffect(() => {
    if (isMobile) {
      const headerElement = document.getElementById('main-header');
      if (headerElement) {
        setHeaderHeight(headerElement.offsetHeight);
      }
    } else {
      setHeaderHeight(0);
    }
  }, [isMobile]);

  const handleSwipe = async (direction: 'left' | 'right') => {
    if (direction === 'right') {
      if (activeCard) {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          const { error } = await supabase.from('likes').insert({
            user_id: user.id,
            video_id: activeCard.id,
          });
          if (error) {
            console.error('Error inserting like:', error);
          }
        }
      }
    }
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const activeCard = activeIndex < cards.length ? cards[activeIndex] : null;

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

  const handleCloseHowToUse = () => {
    setShowHowToUse(false);
  };

  return (
    <motion.div
      className="flex flex-col items-center h-screen overflow-hidden"
      style={{ background: currentGradient }}
      transition={{ duration: 0.3 }}
    >
      <Header cardWidth={cardWidth} />
      <main
        className={`flex-grow flex w-full relative ${isMobile ? 'flex-col bg-white h-full' : 'items-center justify-center'}`}
        style={isMobile ? { paddingTop: `${headerHeight}px` } : {}}
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
              />
            )
          ) : (
            <p className="text-white font-bold text-2xl">Loading videos...</p> // ローディング表示
          )}
        </AnimatePresence>
      </main>
      {!isMobile && (
        <footer className="w-full py-8">
          {activeCard && <ActionButtons
            onSkip={() => triggerSwipe('left')}
            onLike={() => triggerSwipe('right')}
            nopeColor="#A78BFA"
            likeColor="#FBBF24"
            cardWidth={cardWidth}
          />}
        </footer>
      )}
      <AnimatePresence>
        {showHowToUse && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            className="fixed bottom-0 left-0 z-50"
          >
            <HowToUseCard onClose={handleCloseHowToUse} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

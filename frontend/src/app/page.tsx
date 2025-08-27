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

// APIから受け取るvideoオブジェクトの型定義
interface VideoFromApi {
  id: number;
  title: string;
  description: string;
  external_id: string;
  thumbnail_url: string;
  sample_video_url?: string;
  product_released_at?: string;
  performers: { id: string; name: string }[];
  tags: { id: string; name: string }[];
}

const ORIGINAL_GRADIENT = 'linear-gradient(to right, #C4C8E3, #D7D1E3, #F7D7E0, #F8DBB9)';
const LEFT_SWIPE_GRADIENT = 'linear-gradient(to right, #AEB4EB, #D7D1E3, #F7D7E0,#F8DBB9)'; // 左端を明るく
const RIGHT_SWIPE_GRADIENT = 'linear-gradient(to right, #C4C8E3,  #D7D1E3, #F7D7E0,#F9CFA0)'; // 右端を明るく

export default function Home() {
  

  const [cards, setCards] = useState<CardData[]>([]); // APIからのデータを保持するstate
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const [showHowToUse, setShowHowToUse] = useState(true);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const { width: windowWidth, height: windowHeight } = useWindowSize();
  const [cardWidth, setCardWidth] = useState<number | undefined>(400); // cardWidthをstateとして管理し、デフォルト値を400に設定
  const videoAspectRatio = 21 / 20;
  

  const activeCard = activeIndex < cards.length ? cards[activeIndex] : null; // activeCard の宣言を移動

  

  // APIから動画データを取得する
  useEffect(() => {
    const fetchVideos = async () => {
      const { data: { session } } = await supabase.auth.getSession();

      const headers: HeadersInit = {};
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      const { data, error } = await supabase.functions.invoke('videos-feed', {
        headers,
      });

      if (error) {
        console.error('Error fetching videos:', error);
        return;
      }
      
      // APIレスポンスをCardData形式に変換
      const fetchedCards: CardData[] = data.map((video: VideoFromApi) => {
        // サンプル動画URLがあれば<video>優先、なければFANZAの埋め込みを使う
        const fanzaEmbedUrl = `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/&autoplay=1&muted=1&playsinline=1`;
        return {
          id: video.id,
          title: video.title,
          genre: video.tags.map((tag: { id: string; name: string }) => tag.name), // `tags`オブジェクトの配列から`name`の配列を生成
          description: video.description,
          videoUrl: fanzaEmbedUrl, // 互換のため残す（未使用でも可）
          sampleVideoUrl: video.sample_video_url, // 追加: 直接再生できる場合に使用
          embedUrl: fanzaEmbedUrl, // 追加: iframe用URL
          thumbnail_url: video.thumbnail_url, // サムネイルURLを追加
          product_released_at: video.product_released_at,
          performers: video.performers, // APIが整形済みの配列を返す
          tags: video.tags, // APIが整形済みの配列を返す
        };
      });

      

      setCards(fetchedCards);
    };

    fetchVideos();
  }, []);

  useEffect(() => {
    if (windowHeight) {
      // PC版: 動画エリアを画面の3/5に変更し、それに合わせてカード横幅を計算
      // モバイルは従来の1/2のまま（MobileVideoLayout使用のため見た目には影響なし）
      const targetVideoHeight = (!isMobile ? windowHeight * (3 / 5) : windowHeight / 2);
      const calculatedCardWidth = targetVideoHeight * videoAspectRatio;
      setCardWidth(calculatedCardWidth);
    }
  }, [windowHeight, videoAspectRatio, isMobile]);

  

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
      }
    }
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
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
              />
            )
          ) : (
            // ローディング表示（サークル型スピナー）
            <div className="flex items-center justify-center w-full h-full">
              <div
                className="w-12 h-12 rounded-full border-4 border-gray-300 border-t-violet-500 animate-spin"
                role="status"
                aria-label="Loading videos"
              />
            </div>
          )}
        </AnimatePresence>
      </main>
      {!isMobile && (
        <footer className="py-8 mx-auto" style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}>
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

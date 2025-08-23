'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";
import HowToUseCard from "@/components/HowToUseCard";
import useMediaQuery from "@/hooks/useMediaQuery";
import MobileVideoLayout from "@/components/MobileVideoLayout";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

const ORIGINAL_GRADIENT = 'linear-gradient(to right, #C4C8E3, #D7D1E3, #F7D7E0, #F8DBB9)';
const LEFT_SWIPE_GRADIENT = 'linear-gradient(to right, #AEB4EB, #D7D1E3, #F7D7E0,#F8DBB9)'; // 左端を明るく
const RIGHT_SWIPE_GRADIENT = 'linear-gradient(to right, #C4C8E3,  #D7D1E3, #F7D7E0,#F9CFA0)'; // 右端を明るく

export default function Home() {
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const [showHowToUse, setShowHowToUse] = useState(true);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const [headerHeight, setHeaderHeight] = useState(0);
  const [videos, setVideos] = useState<CardData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Check if user is logged in
        const { data: { session } } = await supabase.auth.getSession();
        
        let apiData, apiError;
        
        if (session) {
          // Logged in user: get personalized recommendations
          console.log('Fetching personalized recommendations...');
          const response = await supabase.functions.invoke('recommendations', {
            body: { limit: 20, exclude_liked: true }
          });
          apiData = response.data;
          apiError = response.error;
          
          // If recommendations fail, fallback to diverse feed
          if (apiError || !apiData?.recommendations) {
            console.log('Recommendations failed, falling back to diverse feed...');
            const fallbackResponse = await supabase.functions.invoke('feed_explore', {
              body: { limit: 20, offset: 0 }
            });
            apiData = fallbackResponse.data;
            apiError = fallbackResponse.error;
          }
        } else {
          // Not logged in: get diverse feed
          console.log('Fetching diverse video feed...');
          const response = await supabase.functions.invoke('feed_explore', {
            body: { limit: 20, offset: 0 }
          });
          apiData = response.data;
          apiError = response.error;
        }
        
        if (apiError) {
          console.error('API Error:', apiError);
          setError('動画の取得に失敗しました');
          return;
        }
        
        // Handle both recommendations and feed_explore response formats
        const videoList = apiData?.recommendations || apiData?.videos || [];
        
        if (videoList.length > 0) {
          const formattedVideos = videoList.map((video: any) => ({
            ...video,
            videoUrl: video.sample_video_url || video.preview_video_url || '',
          }));
          setVideos(formattedVideos);
          
          // Log recommendation info if available
          if (apiData?.recommendations && session) {
            console.log(`Loaded ${formattedVideos.length} personalized recommendations`);
            if (apiData.fallback) {
              console.log('Using fallback recommendations while user embedding updates');
            }
          }
        } else {
          setError('表示する動画がありません');
        }
        
      } catch (err) {
        console.error('Fetch error:', err);
        setError('動画の取得中にエラーが発生しました');
      } finally {
        setLoading(false);
      }
    };
    
    fetchVideos();
  }, []);

  const handleSwipe = async (direction?: 'left' | 'right') => {
    const currentCard = activeIndex < videos.length ? videos[activeIndex] : null;
    
    // 右スワイプ（いいね）の場合、APIを呼び出す
    if (direction === 'right' && currentCard) {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session) {
          // Add like
          await supabase.functions.invoke('likes', {
            body: { video_id: currentCard.id },
          });
          
          // Update user embedding (non-blocking)
          supabase.functions.invoke('update_user_embedding').catch(error => {
            console.warn('User embedding update failed:', error);
          });
        }
      } catch (error) {
        console.error('Failed to add like:', error);
      }
    }
    
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
  };

  const triggerSwipe = async (direction: 'left' | 'right') => {
    await handleSwipe(direction);
    cardRef.current?.swipe(direction);
  };

  const activeCard = activeIndex < videos.length ? videos[activeIndex] : null;

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
      <Header />
      <main
        className={`flex-grow flex w-full relative ${isMobile ? 'flex-col bg-white h-full' : 'items-center justify-center'}`}
        style={isMobile ? { paddingTop: `${headerHeight}px` } : {}} // headerHeight を使って paddingTop を動的に設定
      >
        <AnimatePresence mode="wait">
          {loading ? (
            <div className="flex items-center justify-center">
              <p className="text-white font-bold text-xl">動画を読み込み中...</p>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center">
              <p className="text-red-500 font-bold text-xl">{error}</p>
            </div>
          ) : activeCard ? (
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
              />
            )
          ) : (
            <p className="text-white font-bold text-2xl">動画がありません</p>
          )}
        </AnimatePresence>
      </main>
      {!isMobile && (
        <footer className="w-full max-w-md mx-auto py-8">
          {activeCard && <ActionButtons
            onSkip={() => triggerSwipe('left')}
            onLike={() => triggerSwipe('right')}
            nopeColor="#A78BFA"
            likeColor="#FBBF24"
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

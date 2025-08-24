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
  const [currentPhase, setCurrentPhase] = useState<'discovery' | 'recommendation'>('discovery');
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchSize, setBatchSize] = useState(20);
  const [isTransitioning, setIsTransitioning] = useState(false);

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
    fetchInitialVideos();
  }, []);
  
  const fetchInitialVideos = async () => {
    try {
      setLoading(true);
      setError(null);
      setCurrentPhase('discovery');
      setBatchProgress(0);
      setBatchSize(20);
      
      // Always start with discovery phase (20 random items)
      console.log('Fetching discovery phase videos...');
      const response = await supabase.functions.invoke('feed_explore', {
        body: { limit: 20 }
      });
      
      if (response.error) {
        console.error('API Error:', response.error);
        setError('動画の取得に失敗しました');
        return;
      }
      
      const videoList = response.data?.videos || [];
      
      if (videoList.length > 0) {
        const formattedVideos = videoList.map((video: any) => ({
          ...video,
          videoUrl: video.sample_video_url || video.preview_video_url || '',
        }));
        setVideos(formattedVideos);
        
        console.log(`Loaded ${formattedVideos.length} discovery videos`);
        if (response.data?.phase === 'discovery') {
          console.log('Discovery phase initiated');
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
  
  const fetchRecommendationVideos = async () => {
    try {
      setLoading(true);
      setError(null);
      setCurrentPhase('recommendation');
      setBatchProgress(0);
      setBatchSize(100);
      
      console.log('Fetching recommendation phase videos...');
      const response = await supabase.functions.invoke('recommendations', {
        body: { limit: 100, exclude_liked: true }
      });
      
      if (response.error) {
        console.error('Recommendations API Error:', response.error);
        setError('推奨動画の取得に失敗しました');
        return;
      }
      
      const videoList = response.data?.recommendations || [];
      
      if (videoList.length > 0) {
        const formattedVideos = videoList.map((video: any) => ({
          ...video,
          videoUrl: video.sample_video_url || video.preview_video_url || '',
        }));
        setVideos(formattedVideos);
        
        console.log(`Loaded ${formattedVideos.length} recommendation videos`);
        if (response.data?.phase === 'recommendation') {
          console.log('Recommendation phase initiated');
        }
      } else {
        setError('推奨動画がありません');
      }
      
    } catch (err) {
      console.error('Recommendations fetch error:', err);
      setError('推奨動画の取得中にエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

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
        }
      } catch (error) {
        console.error('Failed to add like:', error);
      }
    }
    
    // Update progress
    const newProgress = batchProgress + 1;
    setBatchProgress(newProgress);
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
    
    // Check if batch is completed
    if (newProgress >= batchSize) {
      await handleBatchCompletion(newProgress);
    }
  };
  
  const handleBatchCompletion = async (completedItems: number) => {
    console.log(`Batch completed: ${currentPhase} phase, ${completedItems} items`);
    setIsTransitioning(true);
    
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      // Trigger user embedding update with batch completion info
      if (session) {
        console.log(`Updating user embedding after ${currentPhase} batch completion...`);
        await supabase.functions.invoke('update_user_embedding', {
          body: {
            batch_phase: currentPhase,
            batch_size: batchSize,
            completed_items: completedItems
          }
        });
      }
      
      // Transition to next phase
      if (currentPhase === 'discovery') {
        // After discovery phase, move to recommendations
        if (session) {
          console.log('Transitioning to recommendation phase...');
          setActiveIndex(0); // Reset index for new batch
          await fetchRecommendationVideos();
        } else {
          // Non-logged-in users restart discovery phase
          console.log('User not logged in, restarting discovery phase...');
          setActiveIndex(0);
          await fetchInitialVideos();
        }
      } else {
        // After recommendation phase, restart from discovery
        console.log('Recommendation batch completed, restarting discovery phase...');
        setActiveIndex(0);
        await fetchInitialVideos();
      }
      
    } catch (error) {
      console.error('Batch completion handling failed:', error);
    } finally {
      setIsTransitioning(false);
    }
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
        {/* Progress Indicator */}
        {!loading && !error && (
          <div className="absolute top-4 left-4 right-4 z-10">
            <div className="bg-white/20 backdrop-blur-sm rounded-lg p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-white text-sm font-medium">
                  {currentPhase === 'discovery' ? 'ディスカバリーフェーズ' : 'レコメンドフェーズ'}
                </span>
                <span className="text-white text-sm">
                  {batchProgress}/{batchSize}
                </span>
              </div>
              <div className="w-full bg-white/30 rounded-full h-2">
                <div 
                  className="bg-white rounded-full h-2 transition-all duration-300"
                  style={{ width: `${(batchProgress / batchSize) * 100}%` }}
                />
              </div>
              {currentPhase === 'discovery' && (
                <p className="text-white text-xs mt-1 opacity-75">
                  あなたの好みを学習中です。すべての動画を評価してください。
                </p>
              )}
              {currentPhase === 'recommendation' && (
                <p className="text-white text-xs mt-1 opacity-75">
                  あなたにおすすめの動画です。
                </p>
              )}
            </div>
          </div>
        )}
        
        <AnimatePresence mode="wait">
          {loading || isTransitioning ? (
            <div className="flex flex-col items-center justify-center">
              <p className="text-white font-bold text-xl mb-2">
                {isTransitioning ? 'フェーズを切り替え中...' : '動画を読み込み中...'}
              </p>
              {isTransitioning && (
                <p className="text-white text-sm opacity-75">
                  {currentPhase === 'discovery' 
                    ? 'あなたの好みを分析して、パーソナライズド推奨を準備しています...' 
                    : '次のディスカバリーセッションを準備しています...'}
                </p>
              )}
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

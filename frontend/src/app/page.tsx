'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef } from "react";
import { AnimatePresence, motion, PanInfo } from "framer-motion";

// ダミーデータ
const DUMMY_CARDS: CardData[] = [
  { id: 1, title: '【VR】VR専用機材じゃないとダメなんでしょ？って思ってた時期が俺にもありました。', category: '#VR #高画質 #素人', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 2, title: '新人グラビアアイドル！初めての撮影で緊張…！', category: '#新人 #グラビア #アイドル', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 3, title: '会社の美人上司と禁断の社内恋愛', category: '#上司 #OL #ドラマ', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 4, title: 'ギャルで人妻とかいうパワーワード', category: '#ギャル #人妻 #ドキュメンタリー', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 5, title: '田舎で育った純朴な彼女との初体験', category: '#田舎 #純朴 #初体験', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
];

const ORIGINAL_GRADIENT = 'linear-gradient(to right, #C4C8E3, #D7D1E3, #F7D7E0, #F8DBB9)';
const LEFT_SWIPE_GRADIENT = 'linear-gradient(to right, #AEB4EB, #D7D1E3, #F7D7E0,#F8DBB9)'; // 左端を明るく
const RIGHT_SWIPE_GRADIENT = 'linear-gradient(to right, #C4C8E3,  #D7D1E3, #F7D7E0,#F9CFA0)'; // 右端を明るく

export default function Home() {
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);

  const handleSwipe = () => {
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT); // スワイプ完了後、元のグラデーションに戻す
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const activeCard = activeIndex < DUMMY_CARDS.length ? DUMMY_CARDS[activeIndex] : null;

  // ドラッグ中に背景色をリアルタイムで変更
  const handleDrag = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (info.offset.x > 50) { // 右に50px以上ドラッグ
      setCurrentGradient(RIGHT_SWIPE_GRADIENT);
    } else if (info.offset.x < -50) { // 左に50px以上ドラッグ
      setCurrentGradient(LEFT_SWIPE_GRADIENT);
    } else {
      setCurrentGradient(ORIGINAL_GRADIENT);
    }
  };

  // ドラッグ終了時に背景色を最終決定
  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    // スワイプアウトした場合は handleSwipe で ORIGINAL_GRADIENT に戻るので、ここでは何もしない
    // スワイプアウトしなかった場合は、カードが戻るので ORIGINAL_GRADIENT に戻す
    if (Math.abs(info.offset.x) <= 100) { // 閾値を超えなかった場合
      setCurrentGradient(ORIGINAL_GRADIENT);
    }
  };

  return (
    <motion.div 
      className="flex flex-col items-center justify-between min-h-screen p-4 overflow-hidden"
      style={{ background: currentGradient }} // ここで背景色を適用
      transition={{ duration: 0.3 }} // 背景色変化のアニメーション
    >
      <Header />
      <main className="flex-grow flex items-center justify-center w-full relative h-[70vh]">
        <AnimatePresence mode="wait">
          {activeCard ? (
            <SwipeCard 
              ref={cardRef}
              key={activeCard.id}
              cardData={activeCard} 
              onSwipe={handleSwipe}
              onDrag={handleDrag} 
              onDragEnd={handleDragEnd} 
            />
          ) : (
            <p className="text-white font-bold text-2xl">No more cards</p>
          )}
        </AnimatePresence>
      </main>
      <footer className="w-full max-w-md px-4 py-1 mx-auto sticky bottom-0 backdrop-blur-sm">
        {activeCard && <ActionButtons 
          onSkip={() => triggerSwipe('left')} 
          onLike={() => triggerSwipe('right')}
          nopeColor="#AEB4EB"
          likeColor="#F9CFA0"
        />}
      </footer>
    </motion.div>
  );
}
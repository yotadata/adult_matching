'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import AgeVerificationModal from "@/components/AgeVerificationModal"; // インポート
import { useState, useRef, useEffect } from "react"; // useEffect をインポート
import { AnimatePresence, motion, PanInfo } from "framer-motion";

// ダミーデータ
const DUMMY_CARDS: CardData[] = [
  { id: 1, title: '【VR】VR専用機材じゃないとダメなんでしょ？って思ってた時期が俺にもありました。', category: '#VR #高画質 #素人', description: 'これは非常に長い説明文のサンプルです。スクロール機能が正しく実装されているかを確認するために、このテキストはカードの表示領域を超える長さを持つ必要があります。繰り返しになりますが、これはスクロールのテスト用です。これは非常に長い説明文のサンプルです。スクロール機能が正しく実装されているかを確認するために、このテキストはカードの表示領域を超える長さを持つ必要があります。繰り返しになりますが、これはスクロールのテスト用です。これは非常に長い説明文のサンプルです。スクロール機能が正しく実装されているかを確認するために、このテキストはカードの表示領域を超える長さを持つ必要があります。繰り返しになりますが、これはスクロールのテスト用です。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 2, title: '新人グラビアアイドル！初めての撮影で緊張…！', category: '#新人 #グラビア #アイドル', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 3, title: '会社の美人上司と禁断の社内恋愛', category: '#上司 #OL #ドラマ', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 4, title: 'ギャルで人妻とかいうパワーワード', category: '#ギャル #人妻 #ドキュメンタリー', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
  { id: 5, title: '田舎で育った純朴な彼女との初体験', category: '#田舎 #純朴 #初体験', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.youtube.com/embed/k7Kf89f9KAw?autoplay=1&mute=1&loop=1&playlist=k7Kf89f9KAw' },
];

const ORIGINAL_GRADIENT = 'linear-gradient(to right, #C4C8E3, #D7D1E3, #F7D7E0, #F8DBB9)';
const LEFT_SWIPE_GRADIENT = 'linear-gradient(to right, #A78BFA, #D7D1E3, #F7D7E0,#F8DBB9)'; // 左端をビビッドな紫に
const RIGHT_SWIPE_GRADIENT = 'linear-gradient(to right, #C4C8E3,  #D7D1E3, #F7D7E0,#FBBF24)'; // 右端をビビッドなオレンジに

export default function Home() {
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);
  const [currentGradient, setCurrentGradient] = useState(ORIGINAL_GRADIENT);
  const [showModal, setShowModal] = useState(false); // モーダルの表示状態

  // 年齢確認ロジック
  useEffect(() => {
    const isVerified = localStorage.getItem('ageVerified');
    if (isVerified !== 'true') {
      setShowModal(true);
    }
  }, []);

  const handleAgeConfirm = () => {
    localStorage.setItem('ageVerified', 'true');
    setShowModal(false);
  };

  const handleAgeCancel = () => {
    window.location.href = 'https://www.google.com/search?q=cat';
  };

  const handleSwipe = () => {
    setActiveIndex((prev) => prev + 1);
    setCurrentGradient(ORIGINAL_GRADIENT);
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const activeCard = activeIndex < DUMMY_CARDS.length ? DUMMY_CARDS[activeIndex] : null;

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

  return (
    <>
      {showModal && <AgeVerificationModal onConfirm={handleAgeConfirm} onCancel={handleAgeCancel} />}
      <motion.div 
        className="flex flex-col items-center min-h-screen p-4 overflow-hidden"
        style={{ background: currentGradient }}
        transition={{ duration: 0.3 }}
      >
        <Header />
        <main className="flex-grow flex items-center justify-center w-full relative">
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
        <footer className="w-full max-w-md mx-auto pb-4">
          {activeCard && <ActionButtons 
            onSkip={() => triggerSwipe('left')} 
            onLike={() => triggerSwipe('right')}
            nopeColor="#A78BFA"
            likeColor="#FBBF24"
          />}
        </footer>
      </motion.div>
    </>
  );
}

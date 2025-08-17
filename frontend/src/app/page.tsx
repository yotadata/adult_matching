'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef } from "react";
import { AnimatePresence } from "framer-motion";

// ダミーデータ
const DUMMY_CARDS: CardData[] = [
  { id: 1, title: '【VR】VR専用機材じゃないとダメなんでしょ？って思ってた時期が俺にもありました。', category: '#VR #高画質 #素人', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.dmm.co.jp/litevideo/-/part/=/affi_id=rocqe9nnn-001/cid=sone00758/size=1280_720/' },
  { id: 2, title: '新人グラビアアイドル！初めての撮影で緊張…！', category: '#新人 #グラビア #アイドル', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.dmm.co.jp/litevideo/-/part/=/affi_id=rocqe9nnn-001/cid=sone00758/size=1280_720/' },
  { id: 3, title: '会社の美人上司と禁断の社内恋愛', category: '#上司 #OL #ドラマ', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.dmm.co.jp/litevideo/-/part/=/affi_id=rocqe9nnn-001/cid=sone00758/size=1280_720/' },
  { id: 4, title: 'ギャルで人妻とかいうパワーワード', category: '#ギャル #人妻 #ドキュメンタリー', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.dmm.co.jp/litevideo/-/part/=/affi_id=rocqe9nnn-001/cid=sone00758/size=1280_720/' },
  { id: 5, title: '田舎で育った純朴な彼女との初体験', category: '#田舎 #純朴 #初体験', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。', videoUrl: 'https://www.dmm.co.jp/litevideo/-/part/=/affi_id=rocqe9nnn-001/cid=sone00758/size=1280_720/' },
];

export default function Home() {
  const [activeIndex, setActiveIndex] = useState(0);
  const cardRef = useRef<SwipeCardHandle>(null);

  const handleSwipe = () => {
    setActiveIndex((prev) => prev + 1);
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    cardRef.current?.swipe(direction);
  };

  const activeCard = activeIndex < DUMMY_CARDS.length ? DUMMY_CARDS[activeIndex] : null;

  return (
    <div className="flex flex-col items-center justify-between min-h-screen p-4 overflow-hidden">
      <Header />
      <main className="flex-grow flex items-center justify-center w-full relative h-[70vh]">
        <AnimatePresence mode="wait">
          {activeCard ? (
            <SwipeCard 
              ref={cardRef}
              key={activeCard.id}
              cardData={activeCard} 
              onSwipe={handleSwipe}
            />
          ) : (
            <p className="text-white font-bold text-2xl">No more cards</p>
          )}
        </AnimatePresence>
      </main>
      <footer className="p-4">
        {activeCard && <ActionButtons onSkip={() => triggerSwipe('left')} onLike={() => triggerSwipe('right')} />}
      </footer>
    </div>
  );
}
'use client';

import Header from "@/components/Header";
import SwipeCard, { CardData, SwipeCardHandle } from "@/components/SwipeCard";
import ActionButtons from "@/components/ActionButtons";
import { useState, useRef, createRef, RefObject } from "react";
import { AnimatePresence } from "framer-motion";

// ダミーデータ
const DUMMY_CARDS: CardData[] = [
  { id: 5, title: '田舎で育った純朴な彼女との初体験', category: '#田舎 #純朴 #初体験', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。' },
  { id: 4, title: 'ギャルで人妻とかいうパワーワード', category: '#ギャル #人妻 #ドキュメンタリー', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。' },
  { id: 3, title: '会社の美人上司と禁断の社内恋愛', category: '#上司 #OL #ドラマ', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。' },
  { id: 2, title: '新人グラビアアイドル！初めての撮影で緊張…！', category: '#新人 #グラビア #アイドル', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。' },
  { id: 1, title: '【VR】VR専用機材じゃないとダメなんでしょ？って思ってた時期が俺にもありました。', category: '#VR #高画質 #素人', description: 'サンプルテキスト。サンプルテキスト。サンプルテキスト。' },
];

export default function Home() {
  const [cards, setCards] = useState(DUMMY_CARDS);
  const cardRefs = useRef<RefObject<SwipeCardHandle>[]>([]);
  cardRefs.current = cards.map((_, i) => cardRefs.current[i] ?? createRef());

  const handleSwipe = (swipedCardId: number) => {
    setCards((prev) => prev.filter(card => card.id !== swipedCardId));
  };

  const triggerSwipe = (direction: 'left' | 'right') => {
    const topCardIndex = cards.length - 1;
    if (topCardIndex >= 0) {
      const topCardRef = cardRefs.current[topCardIndex];
      topCardRef.current?.swipe(direction);
    }
  };

  return (
    <div className="flex flex-col items-center justify-between min-h-screen p-4 overflow-hidden">
      <Header />
      <main className="flex-grow flex items-center justify-center w-full relative h-[60vh]">
        <AnimatePresence mode="wait">
          {cards.map((card, index) => (
            <SwipeCard 
              ref={cardRefs.current[index]}
              key={card.id} 
              cardData={card} 
              onSwipe={() => handleSwipe(card.id)}
              index={index}
              isTop={index === cards.length - 1}
            />
          ))}
        </AnimatePresence>
        {cards.length === 0 && <p className="text-white font-bold text-2xl">No more cards</p>}
      </main>
      <footer className="p-4">
        <ActionButtons onSkip={() => triggerSwipe('left')} onLike={() => triggerSwipe('right')} />
      </footer>
    </div>
  );
}

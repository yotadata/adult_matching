'use client';

import { useCallback, useMemo, useState } from 'react';

type OnboardingSlidesProps = {
  open: boolean;
  onFinish: () => void;
};

const SLIDES = [
  {
    title: '気になる・気にならないで、AIがあなたの性癖を学習します',
    bullets: [
      '気になった作品に「気になる」をつけるだけ',
      '逆に “そこまで抜けない” ものは「気にならない」扱いでスキップ',
      'AIがあなたの傾向を少しずつ把握していきます',
    ],
  },
  {
    title: '深夜に探し回らなくても、「抜ける候補」を全部ストックしておけます',
    bullets: [
      '気になった作品は全部ストック',
      '後から落ち着いてゆっくり見返せる',
      'あなただけの “抜ける候補リスト” がどんどん溜まっていきます',
    ],
  },
];

export default function OnboardingSlides({ open, onFinish }: OnboardingSlidesProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [touchStartX, setTouchStartX] = useState<number | null>(null);

  const isLastSlide = currentIndex === SLIDES.length - 1;

  const handleNext = useCallback(() => {
    setCurrentIndex((prev) => {
      if (prev >= SLIDES.length - 1) {
        onFinish();
        return prev;
      }
      return prev + 1;
    });
  }, [onFinish]);

  const handlePrimaryAction = useCallback(() => {
    if (isLastSlide) {
      onFinish();
    } else {
      handleNext();
    }
  }, [handleNext, isLastSlide, onFinish]);

  const handleTouchStart = (event: React.TouchEvent) => {
    setTouchStartX(event.touches[0]?.clientX ?? null);
  };

  const handleTouchEnd = (event: React.TouchEvent) => {
    if (touchStartX == null) return;
    const deltaX = event.changedTouches[0]?.clientX ?? touchStartX;
    if (touchStartX - deltaX > 40) {
      handleNext();
    }
    setTouchStartX(null);
  };

  const indicator = useMemo(() => (
    <div className="flex items-center justify-center gap-2">
      {SLIDES.map((_, index) => (
        <span
          key={`slide-dot-${index}`}
          className={`h-2 w-2 rounded-full transition ${index === currentIndex ? 'bg-white' : 'bg-white/30'}`}
        />
      ))}
    </div>
  ), [currentIndex]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center px-4 py-6 bg-slate-950/60 backdrop-blur-sm text-white">
      <div
        className="w-full max-w-md rounded-[32px] border border-white/20 bg-gradient-to-b from-slate-900/95 via-slate-900/90 to-slate-900/80 p-6 sm:p-8 shadow-2xl space-y-6"
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        <div className="space-y-6 text-center">
          <p className="text-[11px] uppercase tracking-[0.4em] text-white/60">Step {currentIndex + 1} / {SLIDES.length}</p>
          <h2 className="text-2xl sm:text-3xl font-extrabold leading-snug">
            {SLIDES[currentIndex].title}
          </h2>
          <ul className="text-left space-y-3 text-base text-white/90">
            {SLIDES[currentIndex].bullets.map((bullet) => (
              <li key={bullet} className="flex gap-3">
                <span className="text-rose-300 mt-1.5">●</span>
                <span>{bullet}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="space-y-4">
          {indicator}
          <button
            type="button"
            onClick={handlePrimaryAction}
            className="w-full rounded-2xl bg-white text-slate-900 font-semibold py-3 text-base shadow-lg shadow-white/30 active:scale-[0.99] transition"
          >
            {isLastSlide ? 'はじめる' : '次へ'}
          </button>
        </div>
      </div>
    </div>
  );
}

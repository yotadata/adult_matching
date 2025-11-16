'use client';

import { RefObject, useEffect, useState } from 'react';

type SpotlightTutorialProps = {
  likeButtonRef: RefObject<HTMLElement>;
  skipButtonRef: RefObject<HTMLElement>;
  visible: boolean;
  onFinish: () => void;
};

type TutorialStep = {
  key: 'like' | 'skip';
  title: string;
  body: string;
};

const STEPS: TutorialStep[] = [
  {
    key: 'like',
    title: 'ここを押すと “ストック” されます。',
    body: '後から見返す “刺さる候補リスト” を作れます。',
  },
  {
    key: 'skip',
    title: 'ここは “そこまで刺さらない” 時に使います。',
    body: 'こういう作品は、おすすめから外れていきます。',
  },
];

export default function SpotlightTutorial({ likeButtonRef, skipButtonRef, visible, onFinish }: SpotlightTutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    if (!visible) return;
    const updateRect = () => {
      const targetRef = (STEPS[currentStep].key === 'like' ? likeButtonRef : skipButtonRef);
      const node = targetRef.current;
      if (!node) {
        setRect(null);
        return;
      }
      setRect(node.getBoundingClientRect());
    };
    updateRect();
    window.addEventListener('resize', updateRect);
    window.addEventListener('scroll', updateRect, true);
    return () => {
      window.removeEventListener('resize', updateRect);
      window.removeEventListener('scroll', updateRect, true);
    };
  }, [currentStep, likeButtonRef, skipButtonRef, visible]);

  useEffect(() => {
    if (!visible) setCurrentStep(0);
  }, [visible]);

  if (!visible) return null;

  const handleAdvance = () => {
    if (currentStep >= STEPS.length - 1) {
      onFinish();
      return;
    }
    setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
  };

  const overlayStyle = (() => {
    if (!rect) return {};
    const radius = Math.max(rect.width, rect.height) / 2 + 32;
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    return {
      background: `radial-gradient(circle at ${centerX}px ${centerY}px, transparent ${radius}px, rgba(15,23,42,0.78) ${radius + 10}px, rgba(15,23,42,0.9) 100%)`,
    };
  })();

  const bubblePositionStyle = (() => {
    if (!rect) {
      return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
    }
    const centerX = rect.left + rect.width / 2;
    const viewportHeight = typeof window !== 'undefined' ? window.innerHeight : 0;
    const preferBelow = viewportHeight ? rect.top < viewportHeight / 2 : true;
    const top = preferBelow ? rect.bottom + 16 : rect.top - 16;
    return {
      top,
      left: centerX,
      transform: preferBelow ? 'translate(-50%, 0)' : 'translate(-50%, -100%)',
    };
  })();

  const buttonLabel = currentStep === STEPS.length - 1 ? 'はじめる' : '次へ';

  return (
    <div className="fixed inset-0 z-[65] pointer-events-none">
      <div className="absolute inset-0 pointer-events-auto transition-colors duration-300" style={overlayStyle} />
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute max-w-xs sm:max-w-sm rounded-2xl bg-white text-slate-900 shadow-2xl p-4 space-y-2 pointer-events-auto"
          style={bubblePositionStyle}
        >
          <div>
            <p className="text-sm font-bold">{STEPS[currentStep].title}</p>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">{STEPS[currentStep].body}</p>
          </div>
          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleAdvance}
              className="px-4 py-1.5 rounded-full bg-slate-900 text-white text-xs font-semibold shadow-md active:scale-[0.98]"
            >
              {buttonLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

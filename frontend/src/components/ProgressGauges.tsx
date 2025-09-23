'use client';

import React from 'react';

type Props = {
  decisionCount: number;
  personalizeTarget?: number; // デフォルト 20
  diagnosisTarget?: number; // デフォルト 30
  widthPx?: number; // カード幅に合わせる
};

const clamp = (n: number, min = 0, max = 100) => Math.max(min, Math.min(max, n));

const ProgressGauges: React.FC<Props> = ({
  decisionCount,
  personalizeTarget = 20,
  diagnosisTarget = 30,
  widthPx,
}) => {
  const nextLabel = (() => {
    if (decisionCount < personalizeTarget) {
      return `あと${personalizeTarget - decisionCount}枚でパーソナライズ`;
    }
    if (decisionCount < diagnosisTarget) {
      return `あと${diagnosisTarget - decisionCount}枚で診断`;
    }
    return '準備完了';
  })();

  const progressPct = clamp((decisionCount / diagnosisTarget) * 100);
  const pTick = clamp((personalizeTarget / diagnosisTarget) * 100);
  const dTick = 100; // 診断のしきい値

  return (
    <div
      className="w-full sm:w-auto max-w-full rounded-none sm:rounded-xl px-3 py-2 bg-white/70 backdrop-blur border-none sm:border sm:border-white/60"
      style={{ width: widthPx ? `${widthPx}px` : undefined }}
      aria-label="スワイプ進捗ゲージ（コンパクト）"
    >
      <div className="flex items-center gap-3 min-w-0">
        {/* Single compact bar */}
        <div className="relative h-2.5 flex-1 rounded-full bg-white/65 border border-white/70 overflow-hidden">
          {/* fill */}
          <div
            className="h-full bg-gradient-to-r from-indigo-300 via-violet-300 to-rose-300"
            style={{ width: `${progressPct}%` }}
          />

          {/* personalize tick */}
          <div
            className="absolute top-[-2px] w-[2px] h-[14px] bg-gray-500/80"
            style={{ left: `calc(${pTick}% - 1px)` }}
            aria-label="パーソナライズ閾値"
            title="パーソナライズ閾値"
          />
          {/* diagnosis tick (end) */}
          <div
            className="absolute top-[-2px] w-[2px] h-[14px] bg-gray-500/80 right-0"
            aria-label="診断閾値"
            title="診断閾値"
          />
        </div>

        {/* Next milestone text */}
        <div className="text-[11px] sm:text-xs text-gray-700 whitespace-nowrap select-none overflow-hidden text-ellipsis">
          {nextLabel}
        </div>
      </div>
    </div>
  );
};

export default ProgressGauges;

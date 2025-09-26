'use client';

import { useState } from 'react';

interface LikertScaleProps {
  onChange: (value: number) => void;
  value: number | undefined;
}

// Nope/Likeボタンの色
const nopeColor = { r: 108, g: 117, b: 125 }; // #6C757D
const likeColor = { r: 255, g: 107, b: 129 }; // #FF6B81

// 2つの色を線形補間する関数
const lerpColor = (color1: {r,g,b}, color2: {r,g,b}, factor: number) => {
  const r = Math.round(color1.r + factor * (color2.r - color1.r));
  const g = Math.round(color1.g + factor * (color2.g - color1.g));
  const b = Math.round(color1.b + factor * (color2.b - color1.b));
  return `rgb(${r}, ${g}, ${b})`;
};

export default function LikertScale({ onChange, value }: LikertScaleProps) {
  const options = Array.from({ length: 7 }, (_, i) => i + 1);

  return (
    <div className="flex items-center justify-between w-full max-w-lg mx-auto">
      <span className="text-sm font-bold" style={{ color: `rgb(${nopeColor.r}, ${nopeColor.g}, ${nopeColor.b})` }}>
        そう思わない
      </span>
      <div className="flex items-center justify-center space-x-2 sm:space-x-3">
        {options.map((optionValue, index) => {
          const size = 20 + (3 - Math.abs(index - 3)) * 4; // 0-6のindexで端が大きく
          const factor = index / (options.length - 1); // 0-1の補間係数
          const color = lerpColor(nopeColor, likeColor, factor);
          const isSelected = value === optionValue;

          return (
            <button
              key={optionValue}
              onClick={() => onChange(optionValue)}
              className="rounded-full transition-all duration-200 flex items-center justify-center"
              style={{
                width: `${size}px`,
                height: `${size}px`,
                backgroundColor: color,
                boxShadow: isSelected ? `0 0 0 4px rgba(255, 255, 255, 0.5)` : 'none',
                transform: isSelected ? 'scale(1.1)' : 'scale(1)',
              }}
              aria-label={`選択肢 ${optionValue}`}
            />
          );
        })}
      </div>
      <span className="text-sm font-bold" style={{ color: `rgb(${likeColor.r}, ${likeColor.g}, ${likeColor.b})` }}>
        そう思う
      </span>
    </div>
  );
}

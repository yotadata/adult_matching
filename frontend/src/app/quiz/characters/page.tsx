import Link from 'next/link';
import Image from 'next/image';
import type { Metadata } from 'next';
import { QUIZ_TYPES, QuizTypeKey } from '../data';

export const metadata: Metadata = {
  title: 'キャラクター一覧 | 性癖16タイプ診断',
  description: '性癖16タイプ診断の16タイプを一覧で紹介します。あなたはどのタイプ？',
};

const TYPE_ORDER: QuizTypeKey[] = [
  'snph', 'snpl', 'sneh', 'snel',
  'sxph', 'sxpl', 'sxeh', 'sxel',
  'mnph', 'mnpl', 'mneh', 'mnel',
  'mxph', 'mxpl', 'mxeh', 'mxel',
];

const GROUP_LABELS = [
  { label: '支配 × 日常', range: [0, 4], color: '#C0392B' },
  { label: '支配 × 非日常', range: [4, 8], color: '#2471A3' },
  { label: '奉仕 × 日常', range: [8, 12], color: '#17A589' },
  { label: '奉仕 × 非日常', range: [12, 16], color: '#7D3C98' },
];

export default function CharactersPage() {
  return (
    <div className="max-w-lg mx-auto px-4 py-10">

      <div className="text-center mb-10">
        <p className="text-[11px] font-black tracking-[0.3em] text-[#b5541a]/50 uppercase mb-2">✦ Characters ✦</p>
        <h1 className="text-3xl font-black text-[#3d1a00] mb-2">キャラクター一覧</h1>
        <p className="text-[14px] text-[#8b5e3c]">全16タイプを紹介します。あなたはどれ？</p>
      </div>

      {GROUP_LABELS.map((group) => {
        const types = TYPE_ORDER.slice(group.range[0], group.range[1]);
        return (
          <div key={group.label} className="mb-10">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex-1 h-px" style={{ background: `${group.color}40` }} />
              <h2 className="text-[12px] font-black tracking-[0.2em]" style={{ color: group.color }}>
                {group.label}
              </h2>
              <div className="flex-1 h-px" style={{ background: `${group.color}40` }} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              {types.map((key) => {
                const t = QUIZ_TYPES[key];
                return (
                  <Link
                    key={key}
                    href={`/quiz/result/${key}`}
                    className="rounded-3xl overflow-hidden transition-transform active:scale-95"
                    style={{
                      background: '#fffdf5',
                      border: '2px solid #e0c090',
                      outline: '2px dashed rgba(180,120,60,0.25)',
                      outlineOffset: '-7px',
                      boxShadow: '0 3px 0 #c8946a, 0 6px 16px rgba(100,50,0,0.10)',
                    }}
                  >
                    {/* カラーヘッダー */}
                    <div
                      className="h-44 flex items-center justify-center relative overflow-hidden"
                      style={{ background: `${t.color}22` }}
                    >
                      <div
                        className="absolute inset-2 rounded-2xl"
                        style={{ background: `${t.color}18`, border: `1.5px dashed ${t.color}60` }}
                      />
                      <div
                        className="absolute bottom-0 left-0 right-0 h-5"
                        style={{ background: '#fffdf5', clipPath: 'ellipse(65% 100% at 50% 100%)' }}
                      />
                      <Image
                        src={`/quiz/${key}.png`}
                        alt={t.name}
                        width={148}
                        height={148}
                        className="relative object-contain"
                        style={{ filter: 'drop-shadow(0 4px 10px rgba(0,0,0,0.15))' }}
                      />
                    </div>

                    {/* テキスト */}
                    <div className="px-3 pt-2.5 pb-3">
                      <div className="flex gap-1 mb-1.5">
                        {key.toUpperCase().split('').map((c, i) => (
                          <span
                            key={i}
                            className="text-[9px] font-black px-1.5 py-0.5 rounded-full"
                            style={{ background: `${t.color}22`, color: t.accent, border: `1px solid ${t.color}50` }}
                          >
                            {c}
                          </span>
                        ))}
                      </div>
                      <p className="text-[15px] font-black text-[#3d1a00] leading-tight mb-1">
                        {t.name}
                      </p>
                      <p className="text-[11px] text-[#8b5e3c] leading-snug line-clamp-2">
                        {t.tagline}
                      </p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* CTA */}
      <Link
        href="/quiz"
        className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] mt-2 active:translate-y-[1px] transition-transform"
        style={{ background: '#c87941', boxShadow: '0 4px 0 #9e5a28', border: '2px solid #e8a060' }}
      >
        自分のタイプを診断する ✦
      </Link>
    </div>
  );
}

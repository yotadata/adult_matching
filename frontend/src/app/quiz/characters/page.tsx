import Link from 'next/link';
import Image from 'next/image';
import type { Metadata } from 'next';
import { QUIZ_TYPES, QuizTypeKey } from '../data';

export const metadata: Metadata = {
  title: 'キャラクター一覧 | 性癖パーソナリティ診断',
  description: '性癖パーソナリティ診断の16タイプを一覧で紹介します。あなたはどのタイプ？',
};

const TYPE_ORDER: QuizTypeKey[] = [
  'dnph', 'dnpl', 'dneh', 'dnel',
  'dxph', 'dxpl', 'dxeh', 'dxel',
  'snph', 'snpl', 'sneh', 'snel',
  'sxph', 'sxpl', 'sxeh', 'sxel',
];

const GROUP_LABELS = [
  { label: '支配（S） × 日常', range: [0, 4], color: '#FF6B6B' },
  { label: '支配（S） × 非日常', range: [4, 8], color: '#FD79A8' },
  { label: '奉仕（M） × 日常', range: [8, 12], color: '#55EFC4' },
  { label: '奉仕（M） × 非日常', range: [12, 16], color: '#A29BFE' },
];

export default function CharactersPage() {
  return (
    <div className="max-w-lg mx-auto px-4 py-10">

      <div className="text-center mb-8">
        <h1 className="text-2xl font-black text-[#5c2e00] mb-2">キャラクター一覧</h1>
        <p className="text-sm text-[#7a4a1a]">全16タイプを紹介します。あなたはどれ？</p>
      </div>

      {GROUP_LABELS.map((group) => {
        const types = TYPE_ORDER.slice(group.range[0], group.range[1]);
        return (
          <div key={group.label} className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-3 w-3 rounded-full" style={{ background: group.color }} />
              <h2 className="text-[11px] font-black tracking-[0.25em] uppercase" style={{ color: group.color }}>
                {group.label}
              </h2>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {types.map((key) => {
                const t = QUIZ_TYPES[key];
                return (
                  <Link
                    key={key}
                    href={`/quiz/result/${key}`}
                    className="rounded-2xl overflow-hidden transition-transform active:scale-95"
                    style={{
                      background: '#fffdf8',
                      boxShadow: `0 2px 0 ${t.accent}55, 0 4px 0 ${t.accent}88, 0 6px 16px rgba(0,0,0,0.08)`,
                    }}
                  >
                    {/* カラーヘッダー */}
                    <div
                      className="h-32 flex items-center justify-center relative"
                      style={{ background: t.color }}
                    >
                      <div
                        className="absolute bottom-0 left-0 right-0 h-4"
                        style={{
                          background: '#fffdf8',
                          clipPath: 'ellipse(60% 100% at 50% 100%)',
                        }}
                      />
                      <Image
                        src={`/quiz/${key}.png`}
                        alt={t.name}
                        width={112}
                        height={112}
                        className="relative object-contain"
                      />
                    </div>

                    {/* テキスト */}
                    <div className="px-3 pt-2 pb-3">
                      <div className="flex gap-1 mb-1.5">
                        {key.toUpperCase().split('').map((c, i) => (
                          <span
                            key={i}
                            className="text-[9px] font-black px-1.5 py-0.5 rounded-full"
                            style={{ background: t.color, color: t.accent }}
                          >
                            {c}
                          </span>
                        ))}
                      </div>
                      <p className="text-[13px] font-black text-[#3d1a00] leading-tight mb-1">
                        {t.name}
                      </p>
                      <p className="text-[11px] text-[#7a4a1a] leading-snug line-clamp-2">
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
        className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] mt-4"
        style={{ background: 'linear-gradient(90deg, #ff6b6b, #ffd93d)', boxShadow: '0 4px 0 #e08020' }}
      >
        自分のタイプを診断する 🔥
      </Link>
    </div>
  );
}

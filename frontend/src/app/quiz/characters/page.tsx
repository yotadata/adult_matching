import Link from 'next/link';
import Image from 'next/image';
import type { Metadata } from 'next';
import { QUIZ_TYPES, QuizTypeKey } from '../data';

export const metadata: Metadata = {
  title: 'キャラクター一覧 | 性癖16タイプ診断',
  description: '性癖16タイプ診断の16タイプを一覧で紹介します。あなたはどのタイプ？',
};

const TYPE_ORDER: QuizTypeKey[] = [
  'snpc', 'snpw', 'snec', 'snew',
  'sxpc', 'sxpw', 'sxec', 'sxew',
  'mnpc', 'mnpw', 'mnec', 'mnew',
  'mxpc', 'mxpw', 'mxec', 'mxew',
];

const GROUP_LABELS = [
  { label: '支配 × 日常', range: [0, 4], color: '#C0392B' },
  { label: '支配 × 非日常', range: [4, 8], color: '#2471A3' },
  { label: '奉仕 × 日常', range: [8, 12], color: '#17A589' },
  { label: '奉仕 × 非日常', range: [12, 16], color: '#7D3C98' },
];

const DARK_CARD = {
  background: 'rgba(28,24,18,0.85)',
  borderRadius: '20px',
  border: '1px solid rgba(180,150,80,0.35)',
  boxShadow: '0 4px 0 rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(180,150,80,0.15)',
  backdropFilter: 'blur(8px)',
};

export default function CharactersPage() {
  return (
    <div className="max-w-lg mx-auto px-4 py-10">

      <div className="text-center mb-10">
        <p className="text-[11px] font-black tracking-[0.3em] uppercase mb-2" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ Characters ✦</p>
        <h1 className="text-3xl font-black mb-2" style={{ color: '#f0e6d3' }}>キャラクター一覧</h1>
        <p className="text-[14px]" style={{ color: 'rgba(200,180,140,0.6)' }}>全16タイプを紹介します。あなたはどれ？</p>
      </div>

      {GROUP_LABELS.map((group) => {
        const types = TYPE_ORDER.slice(group.range[0], group.range[1]);
        return (
          <div key={group.label} className="mb-10">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex-1 h-px" style={{ background: 'rgba(180,150,80,0.2)' }} />
              <h2 className="text-[12px] font-black tracking-[0.2em]" style={{ color: group.color }}>
                {group.label}
              </h2>
              <div className="flex-1 h-px" style={{ background: 'rgba(180,150,80,0.2)' }} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              {types.map((key) => {
                const t = QUIZ_TYPES[key];
                return (
                  <Link
                    key={key}
                    href={`/quiz/result/${key}`}
                    className="rounded-3xl overflow-hidden transition-transform active:scale-95"
                    style={DARK_CARD}
                  >
                    {/* カラーヘッダー */}
                    <div
                      className="h-44 flex items-center justify-center relative overflow-hidden rounded-t-3xl"
                      style={{ background: `${t.color}55` }}
                    >
                      <div
                        className="absolute inset-2 rounded-2xl"
                        style={{ border: `1px solid ${t.color}50` }}
                      />
                      <Image
                        src={`/quiz/${key}.png`}
                        alt={t.name}
                        width={148}
                        height={148}
                        className="relative object-contain"
                        style={{ filter: 'drop-shadow(0 4px 16px rgba(0,0,0,0.5))' }}
                      />
                    </div>

                    {/* テキスト */}
                    <div className="px-3 pt-2.5 pb-3">
                      <div className="flex gap-1 mb-1.5">
                        {key.toUpperCase().split('').map((c, i) => (
                          <span
                            key={i}
                            className="text-[9px] font-black px-1.5 py-0.5 rounded-full"
                            style={{ background: `${t.color}25`, color: t.color, border: `1px solid ${t.color}50` }}
                          >
                            {c}
                          </span>
                        ))}
                      </div>
                      <p className="text-[15px] font-black leading-tight mb-1" style={{ color: '#f0e6d3' }}>
                        {t.name}
                      </p>
                      <p className="text-[11px] leading-snug line-clamp-2" style={{ color: 'rgba(200,180,140,0.55)' }}>
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
        className="block w-full rounded-2xl py-4 text-center font-black text-[15px] mt-2 active:translate-y-[1px] transition-transform"
        style={{ background: 'rgba(180,150,80,0.2)', color: '#e8d5a0', border: '1px solid rgba(180,150,80,0.5)', boxShadow: '0 4px 0 rgba(0,0,0,0.4)' }}
      >
        自分のタイプを診断する ✦
      </Link>
    </div>
  );
}

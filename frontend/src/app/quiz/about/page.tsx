import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'この診断とは | 性癖16タイプ診断',
  description: '性癖16タイプ診断の4つの軸と16タイプについて解説します。',
};

const AXES = [
  {
    label: '支配 ⇄ 奉仕',
    keys: ['S', 'M'],
    colors: ['#FF6B6B', '#55EFC4'],
    desc: '性的な場面でリードしコントロールしたいか、相手に委ねて従うことに快感を感じるか。',
  },
  {
    label: '日常 ⇄ 非日常',
    keys: ['N', 'X'],
    colors: ['#74B9FF', '#FD79A8'],
    desc: '日常のリアルな場面に反応するか、非日常・ファンタジー的な設定に反応するか。',
  },
  {
    label: '快楽 ⇄ 感情',
    keys: ['P', 'E'],
    colors: ['#FDCB6E', '#A29BFE'],
    desc: '身体的・感覚的な刺激で動くか、心のつながりや感情的な高まりで動くか。',
  },
  {
    label: '頻度高 ⇄ 頻度低',
    keys: ['H', 'L'],
    colors: ['#FF8E53', '#DFE6E9'],
    desc: '衝動が頻繁にあり量を求めるか、間を大切にして深さを求めるか。',
  },
];

const STITCH_CARD = {
  background: '#fffdf5',
  borderRadius: '20px',
  border: '2px solid #e0c090',
  outline: '2px dashed rgba(180,120,60,0.35)',
  outlineOffset: '-8px',
  boxShadow: '0 3px 0 #c8946a, 0 6px 16px rgba(100,50,0,0.08)',
};

export default function AboutPage() {
  return (
    <div className="max-w-lg mx-auto px-4 py-10">

      {/* イントロ */}
      <div className="text-center mb-10">
        <p className="text-[11px] font-black tracking-[0.3em] text-[#b5541a]/50 uppercase mb-3">✦ About ✦</p>
        <h1 className="text-3xl font-black text-[#3d1a00] mb-3">この診断とは？</h1>
        <p className="text-[14px] text-[#8b5e3c] leading-relaxed">
          あなたの欲求・興奮のパターンを<br />
          4つの軸から分析する、16タイプ診断です。
        </p>
      </div>

      {/* 4軸の説明 */}
      <div className="mb-10">
        <h2 className="text-[12px] font-black tracking-[0.3em] text-[#b5541a]/50 uppercase mb-4">✦ 4つの診断軸</h2>
        <div className="space-y-3">
          {AXES.map((axis) => (
            <div key={axis.label} className="p-4" style={STITCH_CARD}>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span
                  className="text-[11px] font-black px-2 py-0.5 rounded-full text-white"
                  style={{ background: axis.colors[0] }}
                >
                  {axis.keys[0]}
                </span>
                <span className="text-[14px] font-black text-[#3d1a00]">{axis.label}</span>
                <span
                  className="text-[11px] font-black px-2 py-0.5 rounded-full text-white"
                  style={{ background: axis.colors[1] }}
                >
                  {axis.keys[1]}
                </span>
              </div>
              <p className="text-[13px] text-[#8b5e3c]">{axis.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 16タイプについて */}
      <div className="mb-10">
        <h2 className="text-[12px] font-black tracking-[0.3em] text-[#b5541a]/50 uppercase mb-4">✦ 16タイプについて</h2>
        <div className="p-5" style={STITCH_CARD}>
          <p className="text-[14px] text-[#8b5e3c] leading-relaxed mb-3">
            4軸それぞれで2択に分かれ、<strong className="text-[#3d1a00]">16通りのタイプ</strong>に分類されます。各タイプには童話のキャラクターが対応しており、自分の性癖の構造を客観的に知るきっかけになります。
          </p>
          <p className="text-[14px] text-[#8b5e3c] leading-relaxed">
            診断結果はX（Twitter）やLINEでシェアできます。
          </p>
        </div>
      </div>

      {/* 注意書き */}
      <div
        className="rounded-2xl p-4 mb-10"
        style={{ background: '#fffdf5', border: '2px dashed rgba(180,120,60,0.35)' }}
      >
        <p className="text-[12px] text-[#8b5e3c] leading-relaxed">
          ⚠️ この診断はエンターテインメント目的のものです。医学的・心理学的な診断ではありません。結果はあくまで参考としてお楽しみください。
        </p>
      </div>

      {/* CTA */}
      <Link
        href="/quiz"
        className="block w-full rounded-2xl py-4 text-center font-black text-white text-[15px] active:translate-y-[1px] transition-transform"
        style={{ background: '#c87941', boxShadow: '0 4px 0 #9e5a28', border: '2px solid #e8a060' }}
      >
        さっそく診断してみる ✦
      </Link>
    </div>
  );
}

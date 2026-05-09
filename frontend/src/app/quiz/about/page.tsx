import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'この診断とは | 偏愛16診断',
  description: '偏愛16診断の4つの軸と16タイプについて解説します。',
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
    desc: '慣れ親しんだ相手・場所でも十分に興奮できるか、ホテルや旅先・初対面の緊張感など非日常の刺激がないと物足りないか。',
  },
  {
    label: '快楽 ⇄ 感情',
    keys: ['P', 'E'],
    colors: ['#FDCB6E', '#A29BFE'],
    desc: '身体的・感覚的な刺激で動くか、心のつながりや感情的な高まりで動くか。',
  },
  {
    label: '偏食 ⇄ 雑食',
    keys: ['C', 'W'],
    colors: ['#FF8E53', '#DFE6E9'],
    desc: '興奮する条件が絞られていてズレると冷めるか（偏食）、雰囲気さえ合えば何でも楽しめるか（雑食）。',
  },
];

const DARK_CARD = {
  background: 'rgba(28,24,18,0.85)',
  borderRadius: '20px',
  border: '1px solid rgba(180,150,80,0.35)',
  boxShadow: '0 4px 0 rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(180,150,80,0.15)',
  backdropFilter: 'blur(8px)',
};

export default function AboutPage() {
  return (
    <div className="max-w-lg mx-auto px-4 py-10">

      {/* イントロ */}
      <div className="text-center mb-10">
        <p className="text-[11px] font-black tracking-[0.3em] uppercase mb-3" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ About ✦</p>
        <h1 className="text-3xl font-black mb-3" style={{ color: '#f0e6d3' }}>この診断とは？</h1>
        <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(200,180,140,0.6)' }}>
          あなたの欲求・興奮のパターンを<br />
          4つの軸から分析する、16タイプ診断です。
        </p>
      </div>

      {/* 4軸の説明 */}
      <div className="mb-10">
        <h2 className="text-[12px] font-black tracking-[0.3em] uppercase mb-4" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ 4つの診断軸</h2>
        <div className="space-y-3">
          {AXES.map((axis) => (
            <div key={axis.label} className="p-4" style={DARK_CARD}>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span
                  className="text-[11px] font-black px-2 py-0.5 rounded-full text-white"
                  style={{ background: axis.colors[0] }}
                >
                  {axis.keys[0]}
                </span>
                <span className="text-[14px] font-black" style={{ color: '#f0e6d3' }}>{axis.label}</span>
                <span
                  className="text-[11px] font-black px-2 py-0.5 rounded-full text-white"
                  style={{ background: axis.colors[1] }}
                >
                  {axis.keys[1]}
                </span>
              </div>
              <p className="text-[13px]" style={{ color: 'rgba(200,180,140,0.65)' }}>{axis.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 16タイプについて */}
      <div className="mb-10">
        <h2 className="text-[12px] font-black tracking-[0.3em] uppercase mb-4" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ 16タイプについて</h2>
        <div className="p-5" style={DARK_CARD}>
          <p className="text-[14px] leading-relaxed mb-3" style={{ color: 'rgba(200,180,140,0.65)' }}>
            4軸それぞれで2択に分かれ、<strong style={{ color: '#f0e6d3' }}>16通りのタイプ</strong>に分類されます。各タイプには童話のキャラクターが対応しており、自分の性癖の構造を客観的に知るきっかけになります。
          </p>
          <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(200,180,140,0.65)' }}>
            診断結果はX（Twitter）やLINEでシェアできます。
          </p>
        </div>
      </div>

      {/* 注意書き */}
      <div
        className="rounded-2xl p-4 mb-10"
        style={{ background: 'rgba(28,24,18,0.6)', border: '1px dashed rgba(180,150,80,0.3)' }}
      >
        <p className="text-[12px] leading-relaxed" style={{ color: 'rgba(200,180,140,0.5)' }}>
          ⚠️ この診断はエンターテインメント目的のものです。医学的・心理学的な診断ではありません。結果はあくまで参考としてお楽しみください。
        </p>
      </div>

      {/* CTA */}
      <Link
        href="/quiz"
        className="block w-full rounded-2xl py-4 text-center font-black text-[15px] active:translate-y-[1px] transition-transform"
        style={{ background: 'rgba(180,150,80,0.2)', color: '#e8d5a0', border: '1px solid rgba(180,150,80,0.5)', boxShadow: '0 4px 0 rgba(0,0,0,0.4)' }}
      >
        さっそく診断してみる ✦
      </Link>
    </div>
  );
}

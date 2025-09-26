
// 4つの分析軸
export type Axis = 'sm' | 'daily' | 'physical' | 'frequency';

// 質問の型定義
export interface Question {
  id: number;
  text: string;
  axis: Axis;
  // 回答のスコアへの影響方向 (1: 正方向, -1: 逆方向)
  direction: 1 | -1;
}

// 質問リスト (仮の30問)
export const quizQuestions: Question[] = [
  // S/M軸 (S: +1, M: -1)
  { id: 1, text: '相手を支配することに興奮を覚える', axis: 'sm', direction: 1 },
  { id: 2, text: '束縛されると安心する', axis: 'sm', direction: -1 },
  { id: 3, text: '相手の反応を見るのが好きだ', axis: 'sm', direction: 1 },
  { id: 4, text: '命令されることに喜びを感じる', axis: 'sm', direction: -1 },
  { id: 5, text: '主導権を握りたい', axis: 'sm', direction: 1 },
  { id: 6, text: '受け身でいる方が楽だ', axis: 'sm', direction: -1 },
  { id: 7, text: '少し意地悪なことを試したくなる', axis: 'sm', direction: 1 },

  // 日常/非日常軸 (非日常: +1, 日常: -1)
  { id: 8, text: 'いつもと違う場所や状況に惹かれる', axis: 'daily', direction: 1 },
  { id: 9, text: '安心できるいつものパターンを好む', axis: 'daily', direction: -1 },
  { id: 10, text: 'コスチュームや役割演技に興味がある', axis: 'daily', direction: 1 },
  { id: 11, text: '刺激よりも安定感を重視する', axis: 'daily', direction: -1 },
  { id: 12, text: 'サプライズ的な展開が好きだ', axis: 'daily', direction: 1 },
  { id: 13, text: '予測可能な範囲で楽しみたい', axis: 'daily', direction: -1 },
  { id: 14, text: '非現実的なシチュエーションを空想する', axis: 'daily', direction: 1 },

  // 肉体/精神軸 (肉体: +1, 精神: -1)
  { id: 15, text: '言葉や雰囲気よりも、直接的な接触を重視する', axis: 'physical', direction: 1 },
  { id: 16, text: '相手の感情や心の動きに強く惹かれる', axis: 'physical', direction: -1 },
  { id: 17, text: '身体的な感覚の鋭敏さに自信がある', axis: 'physical', direction: 1 },
  { id: 18, text: '関係性の構築や心理的な駆け引きが好きだ', axis: 'physical', direction: -1 },
  { id: 19, text: '五感で感じる刺激が好きだ', axis: 'physical', direction: 1 },
  { id: 20, text: '知的な会話や共通の趣味が重要だ', axis: 'physical', direction: -1 },
  { id: 21, text: 'スキンシップが多い方が良い', axis: 'physical', direction: 1 },

  // 頻度 低/高 (高: +1, 低: -1)
  { id: 22, text: '性的な探求に多くの時間を費やしたい', axis: 'frequency', direction: 1 },
  { id: 23, text: '量は少なくても、一回一回の質が重要だ', axis: 'frequency', direction: -1 },
  { id: 24, text: '毎日でも性的なことを考えられる', axis: 'frequency', direction: 1 },
  { id: 25, text: '他の趣味や活動も同じくらい大切だ', axis: 'frequency', direction: -1 },
  { id: 26, text: '新しい刺激を常に求めている', axis: 'frequency', direction: 1 },
  { id: 27, text: '一度満足すると、しばらくは落ち着いている', axis: 'frequency', direction: -1 },
  { id: 28, text: '性欲は強い方だと思う', axis: 'frequency', direction: 1 },
  { id: 29, text: 'ムードやタイミングが整わないと気分が乗らない', axis: 'frequency', direction: -1 },
  { id: 30, text: '機会があれば積極的に楽しみたい', axis: 'frequency', direction: 1 },
];

// 16タイプの定義 (仮)
// 命名規則: [S/M][D/U][P/M][L/H]
// S: Sadistic, M: Masochistic
// D: Daily, U: Unusual
// P: Physical, M: Mental
// L: Low-frequency, H: High-frequency
export const personalityTypes: { [key: string]: { title: string; description: string } } = {
  SDPL: { title: 'サディスティックな日常肉体派', description: '仮の説明文です。' },
  SDPH: { title: 'サディスティックな日常肉体派 (高頻度)', description: '仮の説明文です。' },
  SDML: { title: 'サディスティックな日常精神派', description: '仮の説明文です。' },
  SDMH: { title: 'サディスティックな日常精神派 (高頻度)', description: '仮の説明文です。' },
  SUPL: { title: 'サディスティックな非日常肉体派', description: '仮の説明文です。' },
  SUPH: { title: 'サディスティックな非日常肉体派 (高頻度)', description: '仮の説明文です。' },
  SUML: { title: 'サディスティックな非日常精神派', description: '仮の説明文です。' },
  SUMH: { title: 'サディスティックな非日常精神派 (高頻度)', description: '仮の説明文です。' },
  MDPL: { title: 'マゾヒスティックな日常肉体派', description: '仮の説明文です。' },
  MDPH: { title: 'マゾヒスティックな日常肉体派 (高頻度)', description: '仮の説明文です。' },
  MDML: { title: 'マゾヒスティックな日常精神派', description: '仮の説明文です。' },
  MDMH: { title: 'マゾヒスティックな日常精神派 (高頻度)', description: '仮の説明文です。' },
  MUPL: { title: 'マゾヒスティックな非日常肉体派', description: '仮の説明文です。' },
  MUPH: { title: 'マゾヒスティックな非日常肉体派 (高頻度)', description: '仮の説明文です。' },
  MUML: { title: 'マゾヒスティックな非日常精神派', description: '仮の説明文です。' },
  MUMH: { title: 'マゾヒスティックな非日常精神派 (高頻度)', description: '仮の説明文です。' },
};

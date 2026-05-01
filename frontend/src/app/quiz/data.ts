export type Axis = 'ds' | 'nx' | 'pe' | 'hl';
export type QuizTypeKey =
  | 'dnph' | 'dnpl' | 'dneh' | 'dnel'
  | 'dxph' | 'dxpl' | 'dxeh' | 'dxel'
  | 'snph' | 'snpl' | 'sneh' | 'snel'
  | 'sxph' | 'sxpl' | 'sxeh' | 'sxel';

// 5段階スコア: 1=全然違う 〜 5=まさにそう
// reverse=true のとき、スコアは反転（S/X/E/L 寄りを示す質問）
export interface Question {
  id: number;
  axis: Axis;
  text: string;
  reverse: boolean;
}

export interface QuizType {
  key: QuizTypeKey;
  name: string;
  tagline: string;
  description: string;
  color: string;
  accent: string;
  emoji: string;
}

// 各軸のラベル定義（左=A側=高スコア, 右=B側=低スコア）
export const AXIS_META: Record<Axis, {
  labelHigh: string;   // 100%側
  labelLow: string;    // 0%側
  degreesHigh: string[]; // 強〜弱
  degreesLow: string[];  // 強〜弱
  colorHigh: string;
  colorLow: string;
}> = {
  ds: {
    labelHigh: '支配',
    labelLow: '奉仕',
    degreesHigh: ['ドS（支配）', 'S（支配）寄り'],
    degreesLow: ['ドM（奉仕）', 'M（奉仕）寄り'],
    colorHigh: '#FF6B6B',
    colorLow: '#55EFC4',
  },
  nx: {
    labelHigh: '日常',
    labelLow: '非日常',
    degreesHigh: ['超日常派', '日常寄り'],
    degreesLow: ['超ファンタジー派', 'ファンタジー寄り'],
    colorHigh: '#74B9FF',
    colorLow: '#FD79A8',
  },
  pe: {
    labelHigh: '快楽',
    labelLow: '感情',
    degreesHigh: ['超快楽主義', '快楽寄り'],
    degreesLow: ['超感情派', '感情寄り'],
    colorHigh: '#FDCB6E',
    colorLow: '#A29BFE',
  },
  hl: {
    labelHigh: '頻度高',
    labelLow: '頻度低',
    degreesHigh: ['ド頻度高', '頻度高め'],
    degreesLow: ['ド頻度低', '頻度低め'],
    colorHigh: '#FF8E53',
    colorLow: '#DFE6E9',
  },
};

export const QUESTIONS: Question[] = [
  // ── S(サディスト) ⇄ M(マゾヒスト) 軸 ── high=S側, reverse=true → M寄り ──
  { id: 1,  axis: 'ds', text: '性的な場面では、自分がリードする側にいるほうが自然だ',           reverse: false },
  { id: 2,  axis: 'ds', text: '相手を言葉や行動でコントロールできると、興奮が高まる',           reverse: false },
  { id: 3,  axis: 'ds', text: '相手が自分に従ってくれるとき、快感を強く感じる',                 reverse: false },
  { id: 4,  axis: 'ds', text: '「もっとして」と相手に言わせる側でいたい',                       reverse: false },
  { id: 5,  axis: 'ds', text: '相手にコントロールされることで、むしろ解放感がある',              reverse: true  },
  { id: 6,  axis: 'ds', text: 'リードされる立場にいるとき、最も気持ちよくなれる',               reverse: true  },
  { id: 7,  axis: 'ds', text: '相手の指示や要求に応えることに、強い喜びを感じる',               reverse: true  },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── high=N, reverse=true → X寄り ──
  { id: 8,  axis: 'nx', text: '現実にあるような日常的なシーン（職場・自宅・電車など）で興奮しやすい', reverse: false },
  { id: 9,  axis: 'nx', text: 'コスプレや役割設定より、リアルな状況のほうが刺さる',             reverse: false },
  { id: 10, axis: 'nx', text: '日常の延長線上のシーンで、強くドキッとすることがある',           reverse: false },
  { id: 11, axis: 'nx', text: '特別な演出がなくても、ふとしたシチュエーションで欲求が生まれる', reverse: false },
  { id: 12, axis: 'nx', text: 'コスプレ・役割設定など、非日常の演出があると盛り上がる',        reverse: true  },
  { id: 13, axis: 'nx', text: '支配・被支配のような特殊な設定に、強く惹かれる',                 reverse: true  },
  { id: 14, axis: 'nx', text: '現実離れした妄想のシチュエーションで興奮することが多い',         reverse: true  },

  // ── 快楽(P) ⇄ 感情(E) 軸 ── high=P, reverse=true → E寄り ──
  { id: 15, axis: 'pe', text: '身体的な刺激や感覚への反応が強い',                               reverse: false },
  { id: 16, axis: 'pe', text: '視覚・触覚など感覚的な刺激だけで、十分に興奮できる',             reverse: false },
  { id: 17, axis: 'pe', text: '精神的なつながりより、身体の相性を重視することが多い',           reverse: false },
  { id: 18, axis: 'pe', text: '気持ちの盛り上がりがなくても、身体的な刺激があれば十分だ',       reverse: false },
  { id: 19, axis: 'pe', text: '精神的なつながりを感じないと、身体が動かない',                   reverse: true  },
  { id: 20, axis: 'pe', text: '気持ちが盛り上がって初めて、欲求が生まれる',                     reverse: true  },
  { id: 21, axis: 'pe', text: '感情的な盛り上がりがないと、どんな刺激があっても物足りない',     reverse: true  },

  // ── 頻度高(H) ⇄ 頻度低(L) 軸 ── high=H, reverse=true → L寄り ──
  { id: 22, axis: 'hl', text: '性的な欲求は頻繁にある方だと思う',                               reverse: false },
  { id: 23, axis: 'hl', text: '間を空けるより、できるだけ多く楽しみたい',                       reverse: false },
  { id: 24, axis: 'hl', text: '欲求が来たらなかなか抑えられない',                               reverse: false },
  { id: 25, axis: 'hl', text: '毎日でも物足りないと感じることがある',                           reverse: false },
  { id: 26, axis: 'hl', text: '回数より、一度の深さや濃さを大切にしたい',                       reverse: true  },
  { id: 27, axis: 'hl', text: '性的な行為は特別な機会として、じっくり楽しみたい',               reverse: true  },
  { id: 28, axis: 'hl', text: '頻度より質にこだわりたい',                                       reverse: true  },
];

export interface AxisScore {
  pct: number;     // 0〜100（高いほどD/N/P/H寄り）
  label: string;   // 「ドS（支配）」「S（支配）寄り」など
  isHigh: boolean; // true=D/N/P/H側, false=S/X/E/L側
}

export interface QuizResult {
  typeKey: QuizTypeKey;
  scores: Record<Axis, AxisScore>;
}

// 5段階回答(1〜5)から各軸スコアを計算
export function calcResult(answers: Record<number, number>): QuizResult {
  const axes: Axis[] = ['ds', 'nx', 'pe', 'hl'];
  const scores: Record<Axis, AxisScore> = {} as Record<Axis, AxisScore>;

  for (const axis of axes) {
    const qs = QUESTIONS.filter((q) => q.axis === axis);
    const n = qs.length;
    const raw = qs.reduce((sum, q) => {
      const ans = answers[q.id] ?? 3;
      return sum + (q.reverse ? 6 - ans : ans);
    }, 0);
    const pct = Math.round(((raw - n) / (n * 4)) * 100);
    const meta = AXIS_META[axis];
    const isHigh = pct >= 50;

    let label: string;
    if (pct >= 80)       label = meta.degreesHigh[0];
    else if (pct >= 60)  label = meta.degreesHigh[1];
    else if (pct >= 40)  label = 'ニュートラル';
    else if (pct >= 20)  label = meta.degreesLow[1];
    else                 label = meta.degreesLow[0];

    scores[axis] = { pct, label, isHigh };
  }

  const d = scores.ds.isHigh ? 'd' : 's';
  const n = scores.nx.isHigh ? 'n' : 'x';
  const p = scores.pe.isHigh ? 'p' : 'e';
  const h = scores.hl.isHigh ? 'h' : 'l';
  const typeKey = `${d}${n}${p}${h}` as QuizTypeKey;

  return { typeKey, scores };
}

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  dnph: {
    key: 'dnph', name: '肉食系番長', emoji: '🔥',
    tagline: '欲求に素直、ぐいぐい主導権を握るタイプ',
    description: '衝動が来たら止まらない。相手より先に動いて、流れを自分で作る。「もっと」が口癖で、物足りなさを感じると自分からアクションを起こす。リードする側にいると本領発揮するタイプ。',
    color: '#FF6B6B', accent: '#C0392B',
  },
  dnpl: {
    key: 'dnpl', name: 'クールなぐいぐい系', emoji: '🧊',
    tagline: '主導権は渡さない、でも急がない',
    description: '自分がコントロールする側でいたいけど、焦らない。じっくりペースを作って、気づいたら相手が夢中になってる。日常の中でさりげなく主導権を握るのが得意。',
    color: '#4ECDC4', accent: '#1A9B8C',
  },
  dneh: {
    key: 'dneh', name: '情熱系まとめ役', emoji: '❤️‍🔥',
    tagline: '気持ちが高まると、抑えられない',
    description: '感情と欲求がリンクしている。気持ちが盛り上がると行動も加速する。日常の中の積み重ねで興奮するタイプで、「好き」と「もっと」が同時に来る。リードしながらも感情豊か。',
    color: '#FF8E53', accent: '#E74C3C',
  },
  dnel: {
    key: 'dnel', name: 'じわじわくるカリスマ', emoji: '🌙',
    tagline: '寡黙だけど、気づいたら支配されてる',
    description: '多くを語らず、でも確実に場の空気を掌握している。感情的になることは少ないが、その静けさ自体が圧になる。日常のリアルな場面で、じわじわと相手を引き込む。',
    color: '#A29BFE', accent: '#6C5CE7',
  },
  dxph: {
    key: 'dxph', name: '刺激中毒の冒険家', emoji: '⚡',
    tagline: '普通じゃ物足りない、いつも上を求めてる',
    description: '日常の刺激では満足できない。特殊なシチュエーションや非日常的な体験に強く反応する。身体的な刺激への感度が高く、「こんなことしてみたい」リストが尽きない。',
    color: '#FD79A8', accent: '#E84393',
  },
  dxpl: {
    key: 'dxpl', name: '謎多き一匹狼', emoji: '🐺',
    tagline: '自分のペースで、自分だけの世界を持つ',
    description: '欲求はあるが、頻繁には動かない。独自のファンタジーや世界観を持っていて、日常にはない特別な状況にしか本気で反応しない。ミステリアスな雰囲気が自然と出るタイプ。',
    color: '#636E72', accent: '#2D3436',
  },
  dxeh: {
    key: 'dxeh', name: '主役体質のドラマー', emoji: '🎭',
    tagline: '気持ちが動かないと、身体も動かない',
    description: '非日常的なシチュエーションに感情が乗ると、スイッチが入る。頻度よりも「盛り上がり」を重視。特定の設定やシナリオで感情が爆発する、感情トリガー型の欲求持ち。',
    color: '#FDCB6E', accent: '#E17055',
  },
  dxel: {
    key: 'dxel', name: '孤高のロマンチスト', emoji: '🌌',
    tagline: '頻度より深度、滅多に来ないが来たら本物',
    description: 'めったに動かないが、動くときは全力。非日常的な感情体験を重視していて、その瞬間の「深さ」がすべて。ファンタジーな設定で感情が動くと、誰より激しくなる。',
    color: '#6C5CE7', accent: '#4A3AB5',
  },
  snph: {
    key: 'snph', name: '尽くしすぎ注意報', emoji: '🌿',
    tagline: '相手が喜ぶことが、自分の快感になる',
    description: '相手の反応を見るのが何より好き。日常的な場面で、相手の身体的な満足を引き出すことに強い充実感を覚える。欲求の頻度も高く、尽くす回数も多い。気づいたらやりすぎてる。',
    color: '#55EFC4', accent: '#00B894',
  },
  snpl: {
    key: 'snpl', name: '気まぐれ世話焼き', emoji: '🌸',
    tagline: '気が向いたとき、とことん尽くす',
    description: '欲求もサービス精神も、波がある。でもやると決めたら手加減しない。日常の中で相手の身体的な反応を引き出すことに喜びを感じるが、ペースは完全に自分で決める。',
    color: '#81ECEC', accent: '#00CEC9',
  },
  sneh: {
    key: 'sneh', name: '全力おせっかい愛情家', emoji: '💝',
    tagline: '感情と欲求が一体化してる',
    description: '気持ちが動くと欲求も動く。相手の感情状態に敏感で、心のつながりを感じることで火がつく。日常の中で頻繁に感情が揺れ動き、その都度ちゃんと反応してしまう正直な人。',
    color: '#FF7675', accent: '#D63031',
  },
  snel: {
    key: 'snel', name: '縁の下の天使', emoji: '🕊️',
    tagline: '深い感情の共鳴を、静かに待てる人',
    description: '心がつながったと感じるときにだけ、スイッチが入る。頻度より質派で、日常の延長で相手の感情に寄り添うことに快感がある。派手な刺激より、静かな充足を好む。',
    color: '#DFE6E9', accent: '#B2BEC3',
  },
  sxph: {
    key: 'sxph', name: '妄想族の甘やかし屋', emoji: '🍬',
    tagline: '頭の中はファンタジー、でも行動は尽くし型',
    description: '妄想の中に非日常的な設定がたくさんある。でも実際には相手の身体的な満足を引き出すことに全力で、頻繁に動く。ギャップがすごい。「実はこんなこと考えてた」と言われるタイプ。',
    color: '#FAB1A0', accent: '#E17055',
  },
  sxpl: {
    key: 'sxpl', name: '夢見るロマンチスト', emoji: '✨',
    tagline: '特別な場面でしか、本気にならない',
    description: '日常じゃない。特殊なシチュエーションや非現実的な設定に、身体が正直に反応する。頻度は少ないが、その分「理想の場面」へのこだわりが強い。妄想の解像度が異常に高い。',
    color: '#FFF4E6', accent: '#FDCB6E',
  },
  sxeh: {
    key: 'sxeh', name: '愛に生きる感情家', emoji: '💫',
    tagline: '感情の爆発が、欲求のスイッチになる',
    description: '非日常的な状況や特殊な関係性の中で感情が動いたとき、欲求が最大化する。心が揺れるほど欲しくなる、感情連動型。頻度も高く、感情的な盛り上がりを何度も求めてしまう。',
    color: '#FD79A8', accent: '#E84393',
  },
  sxel: {
    key: 'sxel', name: '一途すぎる月の人', emoji: '🌕',
    tagline: '深い感情が動いたとき、唯一スイッチが入る',
    description: 'めったに欲求が来ないが、来たときは深い。非日常的な感情体験との組み合わせでしか本気にならない。心が完全に動いた相手には全力で応じる、一生に数回型の本気タイプ。',
    color: '#C7ECEE', accent: '#74B9FF',
  },
};

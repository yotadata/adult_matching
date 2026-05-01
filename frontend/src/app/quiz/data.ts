export type Axis = 'ds' | 'nx' | 'pe' | 'hl';
export type AxisValue = 'd' | 's' | 'n' | 'x' | 'p' | 'e' | 'h' | 'l';
export type QuizTypeKey =
  | 'dnph' | 'dnpl' | 'dneh' | 'dnel'
  | 'dxph' | 'dxpl' | 'dxeh' | 'dxel'
  | 'snph' | 'snpl' | 'sneh' | 'snel'
  | 'sxph' | 'sxpl' | 'sxeh' | 'sxel';

export interface Question {
  id: number;
  axis: Axis;
  text: string;
  choiceA: { label: string; value: 'a' };
  choiceB: { label: string; value: 'b' };
}

export interface QuizType {
  key: QuizTypeKey;
  name: string;
  tagline: string;
  description: string;
  color: string;       // bg color
  accent: string;      // felt accent color
  emoji: string;
}

export const QUESTIONS: Question[] = [
  {
    id: 1,
    axis: 'ds',
    text: '気になる相手がいるとき、自分はどっちに近い？',
    choiceA: { label: 'こちらからぐいぐいアプローチしたい', value: 'a' },
    choiceB: { label: '相手の出方を待ちながら、合わせていきたい', value: 'b' },
  },
  {
    id: 2,
    axis: 'ds',
    text: 'ふたりきりの時間、どっちが心地いい？',
    choiceA: { label: '自分がリードして場の流れを作りたい', value: 'a' },
    choiceB: { label: '相手に委ねて、求めに応えることに充実感がある', value: 'b' },
  },
  {
    id: 3,
    axis: 'nx',
    text: '見たり読んだりするコンテンツ、どっち系が好き？',
    choiceA: { label: 'リアルな日常の延長線・実際にありそうな設定', value: 'a' },
    choiceB: { label: '非現実的な設定・ファンタジー・特殊なシチュエーション', value: 'b' },
  },
  {
    id: 4,
    axis: 'nx',
    text: '頭の中で浮かびやすい妄想は？',
    choiceA: { label: '現実に近い、日常の延長のシーン', value: 'a' },
    choiceB: { label: '非日常的な設定や、ちょっとありえない展開', value: 'b' },
  },
  {
    id: 5,
    axis: 'pe',
    text: '興奮するとき、主に何が動いてる？',
    choiceA: { label: '身体が反応している・感覚的なもの', value: 'a' },
    choiceB: { label: '気持ちや感情が高まっている・精神的なもの', value: 'b' },
  },
  {
    id: 6,
    axis: 'pe',
    text: 'エッチなシーン、何が一番刺さる？',
    choiceA: { label: '視覚や感触など、身体への刺激', value: 'a' },
    choiceB: { label: '雰囲気・関係性・心理的なやりとり', value: 'b' },
  },
  {
    id: 7,
    axis: 'hl',
    text: '自分の欲求について、正直なところ？',
    choiceA: { label: '頻繁にある・衝動が強い・もっとほしい', value: 'a' },
    choiceB: { label: 'たまに・じっくり楽しみたい・間を大事にしたい', value: 'b' },
  },
  {
    id: 8,
    axis: 'hl',
    text: '理想の満足感はどっち？',
    choiceA: { label: '回数が多いほど満足する', value: 'a' },
    choiceB: { label: '少なくても深く・特別な体験を大切にしたい', value: 'b' },
  },
];

// A=左タイプ, B=右タイプ
// ds: A→D, B→S
// nx: A→N, B→X
// pe: A→P, B→E
// hl: A→H, B→L

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  dnph: {
    key: 'dnph',
    name: '肉食系番長',
    tagline: '欲求に素直、ぐいぐい主導権を握るタイプ',
    description: '衝動が来たら止まらない。相手より先に動いて、流れを自分で作る。「もっと」が口癖で、物足りなさを感じると自分からアクションを起こす。リードする側にいると本領発揮するタイプ。',
    color: '#FF6B6B',
    accent: '#C0392B',
    emoji: '🔥',
  },
  dnpl: {
    key: 'dnpl',
    name: 'クールなぐいぐい系',
    tagline: '主導権は渡さない、でも急がない',
    description: '自分がコントロールする側でいたいけど、焦らない。じっくりペースを作って、気づいたら相手が夢中になってる。日常の中でさりげなく主導権を握るのが得意。',
    color: '#4ECDC4',
    accent: '#1A9B8C',
    emoji: '🧊',
  },
  dneh: {
    key: 'dneh',
    name: '情熱系まとめ役',
    tagline: '気持ちが高まると、抑えられない',
    description: '感情と欲求がリンクしている。気持ちが盛り上がると行動も加速する。日常の中の積み重ねで興奮するタイプで、「好き」と「もっと」が同時に来る。リードしながらも感情豊か。',
    color: '#FF8E53',
    accent: '#E74C3C',
    emoji: '❤️‍🔥',
  },
  dnel: {
    key: 'dnel',
    name: 'じわじわくるカリスマ',
    tagline: '寡黙だけど、気づいたら支配されてる',
    description: '多くを語らず、でも確実に場の空気を掌握している。感情的になることは少ないが、その静けさ自体が圧になる。日常のリアルな場面で、じわじわと相手を引き込む。',
    color: '#A29BFE',
    accent: '#6C5CE7',
    emoji: '🌙',
  },
  dxph: {
    key: 'dxph',
    name: '刺激中毒の冒険家',
    tagline: '普通じゃ物足りない、いつも上を求めてる',
    description: '日常の刺激では満足できない。特殊なシチュエーションや非日常的な体験に強く反応する。身体的な刺激への感度が高く、「こんなことしてみたい」リストが尽きない。',
    color: '#FD79A8',
    accent: '#E84393',
    emoji: '⚡',
  },
  dxpl: {
    key: 'dxpl',
    name: '謎多き一匹狼',
    tagline: '自分のペースで、自分だけの世界を持つ',
    description: '欲求はあるが、頻繁には動かない。独自のファンタジーや世界観を持っていて、日常にはない特別な状況にしか本気で反応しない。ミステリアスな雰囲気が自然と出るタイプ。',
    color: '#636E72',
    accent: '#2D3436',
    emoji: '🐺',
  },
  dxeh: {
    key: 'dxeh',
    name: '主役体質のドラマー',
    tagline: '気持ちが動かないと、身体も動かない',
    description: '非日常的なシチュエーションに感情が乗ると、スイッチが入る。頻度よりも「盛り上がり」を重視。特定の設定やシナリオで感情が爆発する、感情トリガー型の欲求持ち。',
    color: '#FDCB6E',
    accent: '#E17055',
    emoji: '🎭',
  },
  dxel: {
    key: 'dxel',
    name: '孤高のロマンチスト',
    tagline: '頻度より深度、滅多に来ないが来たら本物',
    description: 'めったに動かないが、動くときは全力。非日常的な感情体験を重視していて、その瞬間の「深さ」がすべて。ファンタジーな設定で感情が動くと、誰より激しくなる。',
    color: '#6C5CE7',
    accent: '#4A3AB5',
    emoji: '🌌',
  },
  snph: {
    key: 'snph',
    name: '尽くしすぎ注意報',
    tagline: '相手が喜ぶことが、自分の快感になる',
    description: '相手の反応を見るのが何より好き。日常的な場面で、相手の身体的な満足を引き出すことに強い充実感を覚える。欲求の頻度も高く、尽くす回数も多い。気づいたらやりすぎてる。',
    color: '#55EFC4',
    accent: '#00B894',
    emoji: '🌿',
  },
  snpl: {
    key: 'snpl',
    name: '気まぐれ世話焼き',
    tagline: '気が向いたとき、とことん尽くす',
    description: '欲求もサービス精神も、波がある。でもやると決めたら手加減しない。日常の中で相手の身体的な反応を引き出すことに喜びを感じるが、ペースは完全に自分で決める。',
    color: '#81ECEC',
    accent: '#00CEC9',
    emoji: '🌸',
  },
  sneh: {
    key: 'sneh',
    name: '全力おせっかい愛情家',
    tagline: '感情と欲求が一体化してる',
    description: '気持ちが動くと欲求も動く。相手の感情状態に敏感で、心のつながりを感じることで火がつく。日常の中で頻繁に感情が揺れ動き、その都度ちゃんと反応してしまう正直な人。',
    color: '#FF7675',
    accent: '#D63031',
    emoji: '💝',
  },
  snel: {
    key: 'snel',
    name: '縁の下の天使',
    tagline: '深い感情の共鳴を、静かに待てる人',
    description: '心がつながったと感じるときにだけ、スイッチが入る。頻度より質派で、日常の延長で相手の感情に寄り添うことに快感がある。派手な刺激より、静かな充足を好む。',
    color: '#DFE6E9',
    accent: '#B2BEC3',
    emoji: '🕊️',
  },
  sxph: {
    key: 'sxph',
    name: '妄想族の甘やかし屋',
    tagline: '頭の中はファンタジー、でも行動は尽くし型',
    description: '妄想の中に非日常的な設定がたくさんある。でも実際には相手の身体的な満足を引き出すことに全力で、頻繁に動く。ギャップがすごい。「実はこんなこと考えてた」と言われるタイプ。',
    color: '#FAB1A0',
    accent: '#E17055',
    emoji: '🍬',
  },
  sxpl: {
    key: 'sxpl',
    name: '夢見るロマンチスト',
    tagline: '特別な場面でしか、本気にならない',
    description: '日常じゃない。特殊なシチュエーションや非現実的な設定に、身体が正直に反応する。頻度は少ないが、その分「理想の場面」へのこだわりが強い。妄想の解像度が異常に高い。',
    color: '#FFF4E6',
    accent: '#FDCB6E',
    emoji: '✨',
  },
  sxeh: {
    key: 'sxeh',
    name: '愛に生きる感情家',
    tagline: '感情の爆発が、欲求のスイッチになる',
    description: '非日常的な状況や特殊な関係性の中で感情が動いたとき、欲求が最大化する。心が揺れるほど欲しくなる、感情連動型。頻度も高く、感情的な盛り上がりを何度も求めてしまう。',
    color: '#FD79A8',
    accent: '#E84393',
    emoji: '💫',
  },
  sxel: {
    key: 'sxel',
    name: '一途すぎる月の人',
    tagline: '深い感情が動いたとき、唯一スイッチが入る',
    description: 'めったに欲求が来ないが、来たときは深い。非日常的な感情体験との組み合わせでしか本気にならない。心が完全に動いた相手には全力で応じる、一生に数回型の本気タイプ。',
    color: '#C7ECEE',
    accent: '#74B9FF',
    emoji: '🌕',
  },
};

export function calcType(answers: Record<number, 'a' | 'b'>): QuizTypeKey {
  // 各軸の質問ID
  const axisQuestions: Record<Axis, number[]> = {
    ds: [1, 2],
    nx: [3, 4],
    pe: [5, 6],
    hl: [7, 8],
  };

  const score = (axis: Axis) => {
    const ids = axisQuestions[axis];
    return ids.filter((id) => answers[id] === 'a').length;
  };

  const d = score('ds') >= 1 ? 'd' : 's';
  const n = score('nx') >= 1 ? 'n' : 'x';
  const p = score('pe') >= 1 ? 'p' : 'e';
  const h = score('hl') >= 1 ? 'h' : 'l';

  return `${d}${n}${p}${h}` as QuizTypeKey;
}

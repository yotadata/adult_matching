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
    text: '好きな人との関係、どちらに近い？',
    choiceA: { label: '自分がリードして引っ張りたい', value: 'a' },
    choiceB: { label: '相手に寄り添って支えたい', value: 'b' },
  },
  {
    id: 2,
    axis: 'ds',
    text: 'デートのプランを決めるなら？',
    choiceA: { label: '自分で決めてサプライズしたい', value: 'a' },
    choiceB: { label: '相手の希望を聞いて叶えてあげたい', value: 'b' },
  },
  {
    id: 3,
    axis: 'nx',
    text: '好きな恋愛コンテンツのシーンは？',
    choiceA: { label: '日常のなかでふと目が合う、リアルな場面', value: 'a' },
    choiceB: { label: '特殊な設定や非現実的なドキドキ展開', value: 'b' },
  },
  {
    id: 4,
    axis: 'nx',
    text: '理想の告白シーンは？',
    choiceA: { label: 'いつもの場所で、ふとした瞬間に', value: 'a' },
    choiceB: { label: '非日常的な演出でドラマチックに', value: 'b' },
  },
  {
    id: 5,
    axis: 'pe',
    text: '好きな人への気持ち、どちらが強い？',
    choiceA: { label: '一緒にいると気持ちいい・楽しい、感覚的なもの', value: 'a' },
    choiceB: { label: '心がつながっている・わかってくれる、感情的なもの', value: 'b' },
  },
  {
    id: 6,
    axis: 'pe',
    text: '映画を見て感情が動くのはどんなとき？',
    choiceA: { label: '笑いや興奮で気分が上がるとき', value: 'a' },
    choiceB: { label: 'ストーリーに感情移入して泣くとき', value: 'b' },
  },
  {
    id: 7,
    axis: 'hl',
    text: '好きな人への連絡、理想のペースは？',
    choiceA: { label: '毎日でも足りないくらい頻繁に', value: 'a' },
    choiceB: { label: '必要なときに、じっくり深く', value: 'b' },
  },
  {
    id: 8,
    axis: 'hl',
    text: '恋愛で大切にしていることは？',
    choiceA: { label: '濃密な時間をとにかくたくさん', value: 'a' },
    choiceB: { label: '少ないけど特別な時間を大切に', value: 'b' },
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
    tagline: '本能に正直、ぐいぐい引っ張るタイプ',
    description: '欲望に正直で行動力がある。好きになったら全力で突き進む。誰よりも熱くて、誰よりも正直。そのまっすぐさが人を惹きつける。',
    color: '#FF6B6B',
    accent: '#C0392B',
    emoji: '🔥',
  },
  dnpl: {
    key: 'dnpl',
    name: 'クールなぐいぐい系',
    tagline: '支配的だけどマイペース',
    description: '引っ張る力はあるのに、なぜかクール。自分のペースを崩さないまま相手をリードする。「え、そんなこと考えてたの？」と言わせるタイプ。',
    color: '#4ECDC4',
    accent: '#1A9B8C',
    emoji: '🧊',
  },
  dneh: {
    key: 'dneh',
    name: '情熱系まとめ役',
    tagline: '感情で動くリーダー',
    description: '感情が豊かでパワフル。「好き」という気持ちをそのまま行動に変える。周りを巻き込みながら前に進む、自然とみんなが頼るタイプ。',
    color: '#FF8E53',
    accent: '#E74C3C',
    emoji: '❤️‍🔥',
  },
  dnel: {
    key: 'dnel',
    name: 'じわじわくるカリスマ',
    tagline: '寡黙だけど圧がある',
    description: '言葉は少なめでも存在感がある。時間をかけてじわじわと相手の心を掌握する。気づいたら頭から離れなくなってる、そんな人。',
    color: '#A29BFE',
    accent: '#6C5CE7',
    emoji: '🌙',
  },
  dxph: {
    key: 'dxph',
    name: '刺激中毒の冒険家',
    tagline: '非日常と快楽を追い求める',
    description: 'いつもと違うことが大好き。刺激がないと物足りない。「普通」という言葉が一番苦手で、常に新しい体験を求めて走り続ける。',
    color: '#FD79A8',
    accent: '#E84393',
    emoji: '⚡',
  },
  dxpl: {
    key: 'dxpl',
    name: '謎多き一匹狼',
    tagline: 'ミステリアスで掴めない',
    description: '何を考えているかわからない。それなのになぜか気になる。近づいたと思ったら遠くにいる。この距離感こそが最大の武器。',
    color: '#636E72',
    accent: '#2D3436',
    emoji: '🐺',
  },
  dxeh: {
    key: 'dxeh',
    name: '主役体質のドラマー',
    tagline: '感情的で物語の中を生きてる',
    description: '人生がドラマ。感情の起伏が激しく、いつも何かのクライマックスにいる。周りを巻き込んで物語を作り出す、天然の主人公。',
    color: '#FDCB6E',
    accent: '#E17055',
    emoji: '🎭',
  },
  dxel: {
    key: 'dxel',
    name: '孤高のロマンチスト',
    tagline: '深くて重い、でも憧れる',
    description: '感情の深さが普通じゃない。理想が高くて、妥協できない。ひとりの時間に思索を重ね、出会った相手には全力で向き合う。',
    color: '#6C5CE7',
    accent: '#4A3AB5',
    emoji: '🌌',
  },
  snph: {
    key: 'snph',
    name: '尽くしすぎ注意報',
    tagline: '世話焼きが過ぎて心配されるタイプ',
    description: '気づいたら相手のために動いている。尽くすことが好きで、相手が喜ぶ顔を見るのが何より幸せ。でも自分のことも大切にしてね。',
    color: '#55EFC4',
    accent: '#00B894',
    emoji: '🌿',
  },
  snpl: {
    key: 'snpl',
    name: '気まぐれ世話焼き',
    tagline: '癒し系だけど自分ペース',
    description: '優しいけど、マイペース。やると決めたら全力で尽くすが、ペースは自分で決める。そのバランス感覚が心地いいと言われる。',
    color: '#81ECEC',
    accent: '#00CEC9',
    emoji: '🌸',
  },
  sneh: {
    key: 'sneh',
    name: '全力おせっかい愛情家',
    tagline: '感情全開で愛を届ける',
    description: '愛情表現に手加減がない。感情をぶつけることが愛だと信じている。ときに重いと言われるが、それこそが本物の証拠。',
    color: '#FF7675',
    accent: '#D63031',
    emoji: '💝',
  },
  snel: {
    key: 'snel',
    name: '縁の下の天使',
    tagline: '目立たないけど絶対いてほしい',
    description: '表には出ないけど、いなくなると気づく。静かに、でも確実に相手を支えている。その存在感は離れてから気づくやつ。',
    color: '#DFE6E9',
    accent: '#B2BEC3',
    emoji: '🕊️',
  },
  sxph: {
    key: 'sxph',
    name: '妄想族の甘やかし屋',
    tagline: '夢見がちだけど尽くす',
    description: '頭の中は常にファンタジー。でも好きな相手には現実的なほど甘やかす。夢と現実の両方で生きているような不思議な人。',
    color: '#FAB1A0',
    accent: '#E17055',
    emoji: '🍬',
  },
  sxpl: {
    key: 'sxpl',
    name: '夢見るロマンチスト',
    tagline: '非現実的な愛を静かに求める',
    description: '理想の恋愛が頭から離れない。現実より夢の方がリアルに感じる。いつか「あの人」が現れると信じて、静かに待ち続ける。',
    color: '#FFF4E6',
    accent: '#FDCB6E',
    emoji: '✨',
  },
  sxeh: {
    key: 'sxeh',
    name: '愛に生きる感情家',
    tagline: '恋愛に全感情を注ぐ',
    description: '恋愛が人生のエンジン。好きな人のために感情をフルスロットルで使い切る。「そこまで思えるの？」と驚かれるレベルの本気派。',
    color: '#FD79A8',
    accent: '#E84393',
    emoji: '💫',
  },
  sxel: {
    key: 'sxel',
    name: '一途すぎる月の人',
    tagline: '深くて静か、でも刺さる',
    description: '一度決めたら、ずっと。感情は深海のように静かで、でも底知れない。その一途さに触れた人は忘れられなくなる。',
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

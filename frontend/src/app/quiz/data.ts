export type Axis = 'ds' | 'nx' | 'pe' | 'cw';

export type QuizTypeKey =
  | 'spnc' | 'spnw' | 'senc' | 'senw'
  | 'spxc' | 'spxw' | 'sexc' | 'sexw'
  | 'mpnc' | 'mpnw' | 'menc' | 'menw'
  | 'mpxc' | 'mpxw' | 'mexc' | 'mexw';

// optionA = high側（S/N/P/C）, optionB = low側（M/X/E/W）
// 回答: 1=完全にA 〜 5=完全にB
export interface Question {
  id: number;
  axis: Axis;
  optionA: string;
  optionB: string;
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
    degreesHigh: ['ドS', 'S寄り'],
    degreesLow: ['ドM', 'M寄り'],
    colorHigh: '#FF6B6B',
    colorLow: '#55EFC4',
  },
  nx: {
    labelHigh: '日常',
    labelLow: '非日常',
    degreesHigh: ['超日常派', '日常寄り'],
    degreesLow: ['刺激派', '非日常寄り'],
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
  cw: {
    labelHigh: '偏食',
    labelLow: '雑食',
    degreesHigh: ['ガチ偏食', '偏食寄り'],
    degreesLow: ['なんでもいける', '雑食寄り'],
    colorHigh: '#FF8E53',
    colorLow: '#DFE6E9',
  },
};

export const QUESTIONS: Question[] = [
  // ── 支配(S) ⇄ 奉仕(M) 軸 ── A=S側, B=M側 ──
  { id: 1,  axis: 'ds', optionA: '相手を押さえつけて「やめて」と言わせたい', optionB: '押さえつけられて「やめて」と言いたい' },
  { id: 2,  axis: 'ds', optionA: '相手を泣かせるくらい追い詰めたい',         optionB: '泣かされるくらい追い詰められたい' },
  { id: 3,  axis: 'ds', optionA: '「もっとして」と相手に言わせたい',          optionB: '「もっとして」と自分が言いたい' },

  // ── 快楽(P) ⇄ 感情(E) 軸 ── A=P側, B=E側 ──
  { id: 4,  axis: 'pe', optionA: '終わった後、相手への気持ちがスッと冷める', optionB: '終わった後の方が相手への愛着が増す' },
  { id: 5,  axis: 'pe', optionA: '気持ちよければ相手への感情は関係ない',                  optionB: '好きじゃない相手だと気持ちよくても虚しい' },
  { id: 6,  axis: 'pe', optionA: '会って数分で「この人とやりたい」と思うことがある',       optionB: 'ある程度仲良くなった相手じゃないと欲求が生まれない' },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── A=N側, B=X側 ──
  { id: 7,  axis: 'nx', optionA: '毎日会う相手でも欲求は普通に続く',                    optionB: '旅行先やホテルなど、いつもと違う場所だと明らかに興奮が上がる' },
  { id: 8,  axis: 'nx', optionA: '慣れ親しんだ相手の方が安心して本気になれる',          optionB: '初対面に近い緊張感のある相手の方が燃える' },
  { id: 9,  axis: 'nx', optionA: 'いつもの部屋・いつもの雰囲気で十分テンションが上がる', optionB: '「バレるかも」という状況や、非日常の緊張感がないと物足りない' },

  // ── 偏食(C) ⇄ 雑食(W) 軸 ── A=C側, B=W側 ──
  { id: 10, axis: 'cw', optionA: '見た目の好みがはっきりしている',                              optionB: 'ストライクゾーンが広い' },
  { id: 11, axis: 'cw', optionA: '好きなプレイと苦手なプレイがはっきりしている',                optionB: 'たいていのことは楽しめる' },
  { id: 12, axis: 'cw', optionA: 'ちゃんと気持ちが高まるまでの流れがないとスイッチが入らない',  optionB: '流れはあまり関係なく、わりとすぐ乗れる' },
];

export interface AxisScore {
  pct: number;     // 0〜100（高いほどD/N/P/C寄り）
  label: string;   // 「ドS（支配）」「S（支配）寄り」など
  isHigh: boolean; // true=S/N/P/C側, false=M/X/E/W側
}

export interface QuizResult {
  typeKey: QuizTypeKey;
  scores: Record<Axis, AxisScore>;
}

// 5段階回答(1=完全にA 〜 5=完全にB)から各軸スコアを計算
// A=high側(S/P/N/C), B=low側(M/E/X/W) なので 1→100%, 5→0%
export function calcResult(answers: Record<number, number>): QuizResult {
  const axes: Axis[] = ['ds', 'nx', 'pe', 'cw'];
  const scores: Record<Axis, AxisScore> = {} as Record<Axis, AxisScore>;

  for (const axis of axes) {
    const qs = QUESTIONS.filter((q) => q.axis === axis);
    const n = qs.length;
    // ans=1→high寄り(5点換算), ans=5→low寄り(1点換算)
    const raw = qs.reduce((sum, q) => {
      const ans = answers[q.id] ?? 3;
      return sum + (6 - ans);
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

  const d = scores.ds.isHigh ? 's' : 'm';
  const n = scores.nx.isHigh ? 'n' : 'x';
  const p = scores.pe.isHigh ? 'p' : 'e';
  const c = scores.cw.isHigh ? 'c' : 'w';
  const typeKey = `${d}${p}${n}${c}` as QuizTypeKey;

  return { typeKey, scores };
}

// 相性の良いタイプ: ds軸が逆（S↔M）で他の軸が近いほど相性◎
// [最高相性, 高相性, 良相性] の順
export const COMPATIBILITY: Record<QuizTypeKey, [QuizTypeKey, QuizTypeKey, QuizTypeKey]> = {
  spnc: ['mpnc', 'mpnw', 'menc'],
  spnw: ['mpnw', 'mpnc', 'menw'],
  senc: ['menc', 'menw', 'mpnc'],
  senw: ['menw', 'menc', 'mpnw'],
  spxc: ['mpxc', 'mpxw', 'mexc'],
  spxw: ['mpxw', 'mpxc', 'mexw'],
  sexc: ['mexc', 'mexw', 'mpxc'],
  sexw: ['mexw', 'mexc', 'mpxw'],
  mpnc: ['spnc', 'spnw', 'senc'],
  mpnw: ['spnw', 'spnc', 'senw'],
  menc: ['senc', 'senw', 'spnc'],
  menw: ['senw', 'senc', 'spnw'],
  mpxc: ['spxc', 'spxw', 'sexc'],
  mpxw: ['spxw', 'spxc', 'sexw'],
  mexc: ['sexc', 'sexw', 'spxc'],
  mexw: ['sexw', 'sexc', 'spxw'],
};

export const COMPATIBILITY_LABELS: [string, string, string] = ['最高相性', '高相性', '良相性'];

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  spnc: {
    key: 'spnc', name: 'グルメな狩人', emoji: '🔥',
    tagline: '条件が揃ったときだけ、本気になる',
    description: 'リードするのが自然で、外見や身体の相性で相手を選ぶ。慣れた相手・いつもの場所でも十分スイッチが入るが、こだわりの条件が細かくてズレると一気に冷める。「これじゃないと」が多い分、揃ったときの集中力は誰より高い。',
    color: '#FF6B6B', accent: '#C0392B',
  },
  spnw: {
    key: 'spnw', name: '全方位ハンター', emoji: '🧊',
    tagline: '誰でも落とせる、どこでも本気',
    description: '外見や身体の相性で動き、リードするのが好き。慣れた相手でも新しい相手でも十分燃えるし、細かいこだわりもない。「まあなんでもいけるし、リードさせてくれれば」という間口の広さが武器。',
    color: '#4ECDC4', accent: '#1A9B8C',
  },
  senc: {
    key: 'senc', name: 'むっつり帝王', emoji: '❤️‍🔥',
    tagline: '感情のスイッチが入ると、誰より熱い',
    description: '感情が動いた相手をリードしたくなる。慣れ親しんだ日常の中でもしっかり燃えるが、感情のトリガーが細かい。「この人のこういうところ」が揃わないと本気になれず、条件が揃った瞬間は誰より熱くなる。',
    color: '#FF8E53', accent: '#E74C3C',
  },
  senw: {
    key: 'senw', name: '天然の王様', emoji: '🌙',
    tagline: '気持ちが乗れば、どこでも自然にリードする',
    description: '感情が動けばどんな状況でもリードできる。慣れた相手・場所でも十分OKで、細かいこだわりもない。その場の雰囲気を読んで自然にリードし、気づいたら相手が夢中になっている。一番「普通に最強」なタイプ。',
    color: '#A29BFE', accent: '#6C5CE7',
  },
  spxc: {
    key: 'spxc', name: '刺激ジャンキー', emoji: '⚡',
    tagline: '非日常の緊張感の中でしか、本気にならない',
    description: 'ホテルや旅先、バレるかもしれない場所など、いつもと違う刺激がないと物足りない。外見重視でリードしたいが、こだわりが細かく設定がズレると冷める。「いつもと違う場所で、自分のペースで支配する」が理想形。',
    color: '#FD79A8', accent: '#E84393',
  },
  spxw: {
    key: 'spxw', name: 'アドレナリン番長', emoji: '🎲',
    tagline: '非日常さえあれば、なんでもいける',
    description: 'いつもの場所では物足りない。ホテルでも旅先でも、非日常の空気さえあればこだわりなく全力になれる。外見重視でリードするのが好き。「普通の状況はつまらないが、どこでも遊べる」という自由人な支配タイプ。',
    color: '#636E72', accent: '#2D3436',
  },
  sexc: {
    key: 'sexc', name: '過激な演出家', emoji: '🎭',
    tagline: '感情も場所も、全部揃って初めて動く',
    description: '感情が動いた相手をリードしたいが、非日常の場でないとスイッチが入らない。さらに感情のトリガーも細かい。条件が二重に絞られているぶん、全部揃ったときの爆発力はすさまじい。「なんか違う」が多い代わりに、刺さったときは全力。',
    color: '#FDCB6E', accent: '#E17055',
  },
  sexw: {
    key: 'sexw', name: '刺激の旅人', emoji: '🌌',
    tagline: '感情さえ動けば、非日常の中でどこでもリードする',
    description: '感情が動いた相手には強くリードしたくなる。ただ、いつもと同じ場所では物足りない。ホテルや旅先など非日常の空気の中で感情に火がつくと、こだわりなく全力になれる。感情主導で刺激を求めるロマンチストな支配者。',
    color: '#6C5CE7', accent: '#4A3AB5',
  },
  mpnc: {
    key: 'mpnc', name: 'こだわり沼らせ魔', emoji: '🌿',
    tagline: '条件が揃った相手にだけ、全力で尽くす',
    description: '相手を喜ばせることが好きで、外見や身体の相性で相手を選ぶ。慣れた環境でも十分燃えるが、こだわりの条件が細かい。「この人だ」と思えた相手への集中力は異常で、気づいたら相手が深みにはまっている。',
    color: '#55EFC4', accent: '#00B894',
  },
  mpnw: {
    key: 'mpnw', name: '世話焼き八方美人', emoji: '🌸',
    tagline: '誰に対しても、どこでも尽くせる',
    description: '相手の身体的な満足を引き出すのが好きで、こだわりは少ない。慣れた相手でも新しい相手でも自分なりに尽くせる。「まあ誰でもいけるし、喜んでくれれば嬉しい」という、間口の広い頼もしい奉仕タイプ。',
    color: '#F9E784', accent: '#F1C40F',
  },
  menc: {
    key: 'menc', name: 'むっつり溺愛体質', emoji: '💝',
    tagline: 'この感情、この人だけに全部渡す',
    description: '心がつながった相手に尽くしたくなるが、感情のトリガーが細かい。「この人のこういうところが好き」が揃って初めてスイッチが入る。慣れ親しんだ関係の中で愛情が深まるタイプで、刺さると溺愛モードに入る。',
    color: '#E84393', accent: '#C0136F',
  },
  menw: {
    key: 'menw', name: '一途な忠犬', emoji: '🕊️',
    tagline: '気持ちさえ動けば、どんな相手も受け止める',
    description: '心がつながったと感じれば、どんな状況でも相手に寄り添える。慣れた日常の中でじっくり愛情が育つタイプで、こだわりなく相手に合わせられる。静かで包容力があり、「あなたに合わせます」という姿勢が自然体。',
    color: '#DFE6E9', accent: '#B2BEC3',
  },
  mpxc: {
    key: 'mpxc', name: '指示多めのしもべ', emoji: '🍬',
    tagline: '非日常の場で、決まった条件が揃って初めて動く',
    description: 'ホテルや旅先など非日常の環境で、外見が好みの相手に尽くすのが理想。こだわりの条件が細かく、設定がズレると冷める。「この場所で、この人に、こういうふうに尽くしたい」という妄想がクリアに決まっている。',
    color: '#FAB1A0', accent: '#E17055',
  },
  mpxw: {
    key: 'mpxw', name: 'スキンシップ魔人', emoji: '✨',
    tagline: 'いつもと違う場所さえあれば、なんでも尽くせる',
    description: '日常では物足りない。ホテルでも旅先でも、非日常の空気さえあれば相手の身体的な満足を引き出すことに全力になれる。細かいこだわりは薄く、「非日常さえあれば何でもいける」という柔軟な刺激追求型の尽くし屋。',
    color: '#FFF4E6', accent: '#FDCB6E',
  },
  mexc: {
    key: 'mexc', name: 'こじらせの極み', emoji: '💫',
    tagline: '感情も場所も揃って、初めて全部渡せる',
    description: '心が動いた相手に尽くしたいが、非日常の場でないとスイッチが入らない。さらに感情のトリガーも細かい。条件が二重に絞られているぶん、全部揃ったときの没入度はすさまじい。旅先やホテルで感情が爆発する愛情家。',
    color: '#E17055', accent: '#C0392B',
  },
  mexw: {
    key: 'mexw', name: '夢見る妄想家', emoji: '🌕',
    tagline: '感情さえ動けば、どんな非日常でも全力で尽くす',
    description: '感情が動いた相手には全力で尽くしたくなる。いつもの場所より非日常の空気の中の方が燃える。細かいこだわりは薄く、ホテルでも旅先でも感情スイッチが入れば何でもOK。感情主導で動く、刺激を求める献身家。',
    color: '#C7ECEE', accent: '#74B9FF',
  },
};

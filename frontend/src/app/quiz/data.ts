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
  { id: 4,  axis: 'pe', optionA: '顔とスタイルが好みなら、性格や関係性は関係なく惹かれる', optionB: 'どれだけ好みでも、気持ちが入らない相手とは無理' },
  { id: 5,  axis: 'pe', optionA: '気持ちよければ相手への感情は関係ない',                  optionB: '好きじゃない相手だと気持ちよくても虚しい' },
  { id: 6,  axis: 'pe', optionA: '会って数分で「この人とやりたい」と思うことがある',       optionB: 'ある程度仲良くなった相手じゃないと欲求が生まれない' },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── A=N側, B=X側 ──
  { id: 7,  axis: 'nx', optionA: '普通の部屋・普段着のシチュで十分興奮できる',          optionB: 'コスプレや特殊な設定があるほうが断然盛り上がる' },
  { id: 8,  axis: 'nx', optionA: 'リアルな状況のほうが刺さる',                          optionB: '現実ではあり得ない設定のほうが興奮する' },
  { id: 9,  axis: 'nx', optionA: '普段の何気ない場面でも欲求が生まれる',                optionB: 'ファンタジー・異世界・特殊な役割設定に強く惹かれる' },

  // ── 偏食(C) ⇄ 雑食(W) 軸 ── A=C側, B=W側 ──
  { id: 10, axis: 'cw', optionA: 'こだわりの条件がちょっとでもズレると一気に冷める',           optionB: '雰囲気さえ合えば細かい条件は全然気にしない' },
  { id: 11, axis: 'cw', optionA: '自分の性癖を友達に話すと「細かすぎ」って引かれる自覚がある', optionB: 'こだわりはほぼなく、わりと何でも楽しめる' },
  { id: 12, axis: 'cw', optionA: '頭の中に理想の展開があって、そこから外れると萎える',          optionB: 'その場の流れに乗れれば何でも楽しい' },
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

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  spnc: {
    key: 'spnc', name: '肉食系番長', emoji: '🔥',
    tagline: '型が決まってる、だから強い',
    description: '自分のやり方とペースがある。リードする側が自然だが、相手や状況が合わないとスイッチが入らない。「これじゃなきゃダメ」な条件がはっきりしていて、それが揃ったときだけ本領発揮する。',
    color: '#FF6B6B', accent: '#C0392B',
  },
  spnw: {
    key: 'spnw', name: '確実に落とす職人', emoji: '🧊',
    tagline: '誰でも落とせる、それが強さ',
    description: '自分がリードする側でいたいが、こだわりは薄い。相手が誰でも、状況がどこでも、自分のペースで場を作れる。「まあなんでもいける」という柔軟さが武器で、気づいたら相手が夢中になってる。',
    color: '#4ECDC4', accent: '#1A9B8C',
  },
  senc: {
    key: 'senc', name: '激情型リーダー', emoji: '❤️‍🔥',
    tagline: '火種が合えば、誰より熱くなる',
    description: '感情とリードがリンクしている。ただし、特定の感情トリガーが揃わないとスイッチが入らない。「この感じじゃないと」という条件がある。刺さった瞬間は誰より熱く、日常の中でそのポイントを探している。',
    color: '#FF8E53', accent: '#E74C3C',
  },
  senw: {
    key: 'senw', name: 'じわじわくるカリスマ', emoji: '🌙',
    tagline: '空気に乗って、気づいたら支配してる',
    description: '感情が乗れば何でもいける。特定のシチュや条件にこだわらず、その場の雰囲気を読んで自然にリードする。「なんとなく好き」が欲求に直結するタイプで、場を選ばずじわじわ引き込む。',
    color: '#A29BFE', accent: '#6C5CE7',
  },
  spxc: {
    key: 'spxc', name: '刺激中毒の冒険家', emoji: '⚡',
    tagline: '決まったシチュでしか本気にならない',
    description: '非日常的な設定に強く反応するが、どんなファンタジーでもいいわけじゃない。「これだ」と思えるシチュエーションが決まっていて、そこから外れると冷める。こだわりリストが細かすぎて、自分でも驚くことがある。',
    color: '#FD79A8', accent: '#E84393',
  },
  spxw: {
    key: 'spxw', name: '全部乗りの冒険家', emoji: '🎲',
    tagline: '非日常なら、なんでもいける',
    description: '普通じゃ物足りないが、特定のシチュへのこだわりはない。コスプレでも役割設定でも異世界でも、非日常っぽければ大体乗れる。「まあ何でも楽しめるけど、普通はつまらない」がモットー。',
    color: '#636E72', accent: '#2D3436',
  },
  sexc: {
    key: 'sexc', name: '劇場型支配者', emoji: '🎭',
    tagline: '脚本が決まって初めてスイッチが入る',
    description: '非日常の感情体験にしか反応しないうえ、そのシナリオが自分の中で決まっている。設定と感情、両方が揃って初めて本気になれる。「お膳立てが合わないと無理」と自覚している、二重こだわり型。',
    color: '#FDCB6E', accent: '#E17055',
  },
  sexw: {
    key: 'sexw', name: '孤高のロマンチスト', emoji: '🌌',
    tagline: '感情さえ乗れば、どんな世界でも入れる',
    description: '非日常な感情体験を求めているが、どんな設定でも感情が動けばOK。ファンタジーな空気さえあれば細かい条件にはこだわらない。感情スイッチが入った瞬間は誰より没入し、その深さがすべてになる。',
    color: '#6C5CE7', accent: '#4A3AB5',
  },
  mpnc: {
    key: 'mpnc', name: '献身体質の沼らせ屋', emoji: '🌿',
    tagline: '刺さる相手にしか、全力を出せない',
    description: '相手が喜ぶことが快感だが、誰にでも尽くせるわけじゃない。「この人だ」と思える条件が揃った相手にしか本気になれない。その分、スイッチが入った相手への集中度は異常で、気づいたら沼らせている。',
    color: '#55EFC4', accent: '#00B894',
  },
  mpnw: {
    key: 'mpnw', name: '気まぐれ世話焼き', emoji: '🌸',
    tagline: '誰でも尽くせる、それが自分のスタイル',
    description: '相手の身体的な満足を引き出すことが好きで、こだわりは少ない。誰が相手でも、どんな状況でも自分なりに尽くせる。「まあなんでもいけるし、喜んでくれれば嬉しい」という、とにかく柔軟な奉仕型。',
    color: '#81ECEC', accent: '#00CEC9',
  },
  menc: {
    key: 'menc', name: '溺愛体質の愛情家', emoji: '💝',
    tagline: 'この感じ、この人だけに全部渡す',
    description: '気持ちが動くと尽くしたくなるが、感情のトリガーが細かい。「この人のこういうところが好き」という条件が揃って初めてスイッチが入る。刺さると溺愛モードに入り、細部まで尽くすこだわり派の愛情家。',
    color: '#FF7675', accent: '#D63031',
  },
  menw: {
    key: 'menw', name: '縁の下の天使', emoji: '🕊️',
    tagline: '気持ちさえ動けば、何でも受け止める',
    description: '心がつながったと感じれば、どんな状況でも相手に寄り添える。シチュや条件にこだわりはなく、感情が動いた相手ならなんでもOK。静かで包容力があり、「あなたに合わせます」という雑食型の献身家。',
    color: '#DFE6E9', accent: '#B2BEC3',
  },
  mpxc: {
    key: 'mpxc', name: '妄想族の甘やかし屋', emoji: '🍬',
    tagline: '頭の中の設定通りじゃないと動けない',
    description: '非日常的な妄想の中に「決まったシチュ」がある。その設定が揃わないと本気になれないが、揃った瞬間に全力で尽くす。「実はこういう場面でこういう展開じゃないとダメ」という細かいこだわりを持つ尽くし屋。',
    color: '#FAB1A0', accent: '#E17055',
  },
  mpxw: {
    key: 'mpxw', name: '解像度の低い妄想家', emoji: '✨',
    tagline: '非日常っぽければ、細かいことは気にしない',
    description: '日常じゃない設定が好きで、身体的な反応に素直。でも「このシチュじゃなきゃ」というこだわりは薄く、非日常感さえあれば大体乗れる。妄想はするが解像度は低め。雰囲気重視のフワッとした尽くし屋。',
    color: '#FFF4E6', accent: '#FDCB6E',
  },
  mexc: {
    key: 'mexc', name: '愛に生きる感情家', emoji: '💫',
    tagline: 'この設定、この感情、全部揃って初めて動く',
    description: '非日常的な感情体験にしか反応しないうえ、その「感情の種類」まで決まっている。二重に条件が絞られていて、揃ったときの爆発力はすさまじい。「なんか違う」が多いが、刺さったときは全部渡せる感情家。',
    color: '#FD79A8', accent: '#E84393',
  },
  mexw: {
    key: 'mexw', name: '一途すぎる月の人', emoji: '🌕',
    tagline: '感情さえ動けば、どんな世界でも全部渡す',
    description: '非日常的な感情体験が好きだが、どんな設定でも感情が動けばOK。細かいシチュへのこだわりはなく、心が動いた相手に対してはどんな状況でも全力で応じる。感情主導で動く、雑食型の一途な献身家。',
    color: '#C7ECEE', accent: '#74B9FF',
  },
};

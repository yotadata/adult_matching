export type Axis = 'ds' | 'nx' | 'pe' | 'cw';

export type QuizTypeKey =
  | 'spnc' | 'spnw' | 'senc' | 'senw'
  | 'spxc' | 'spxw' | 'sexc' | 'sexw'
  | 'mpnc' | 'mpnw' | 'menc' | 'menw'
  | 'mpxc' | 'mpxw' | 'mexc' | 'mexw';

// 5段階スコア: 1=全然違う 〜 5=まさにそう
// reverse=true のとき、スコアは反転（S/X/E/W 寄りを示す質問）
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
    degreesHigh: ['ガチ支配', '支配寄り'],
    degreesLow: ['ガチ奉仕', '奉仕寄り'],
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
  // ── 支配(S) ⇄ 奉仕(M) 軸 ── high=S側, reverse=true → M寄り ──
  { id: 1,  axis: 'ds', text: '性的な場面では、自分がリードする側にいるほうが自然だ',           reverse: false },
  { id: 2,  axis: 'ds', text: '相手を言葉や行動でコントロールできると、興奮が高まる',           reverse: false },
  { id: 3,  axis: 'ds', text: '「もっとして」と相手に言わせる側でいたい',                       reverse: false },
  { id: 4,  axis: 'ds', text: '相手に思い通りにされることで、気持ちが高まる',                    reverse: true  },
  { id: 5,  axis: 'ds', text: '相手の指示や要求に応えることに、強い喜びを感じる',               reverse: true  },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── high=N, reverse=true → X寄り ──
  { id: 6,  axis: 'nx', text: 'コスプレや特殊な役割設定がなくても、現実にありそうな場面で十分に興奮できる', reverse: false },
  { id: 7,  axis: 'nx', text: 'コスプレや役割設定より、リアルな状況のほうが刺さる',             reverse: false },
  { id: 8,  axis: 'nx', text: '特別な場所や雰囲気がなくても、普段の何気ない場面で欲求が生まれる', reverse: false },
  { id: 9,  axis: 'nx', text: 'コスプレ・役割設定など、非日常の演出があると盛り上がる',         reverse: true  },
  { id: 10, axis: 'nx', text: 'コスプレや異世界・特殊な役割など、現実にはない設定に強く惹かれる',  reverse: true  },

  // ── 快楽(P) ⇄ 感情(E) 軸 ── high=P, reverse=true → E寄り ──
  { id: 11, axis: 'pe', text: '外見や雰囲気が好みなら、深く知らなくても欲求が生まれる',         reverse: false },
  { id: 12, axis: 'pe', text: '身体的な相性が合う相手なら、気持ちが深くなくても満足できる',     reverse: false },
  { id: 13, axis: 'pe', text: '気持ちより身体の感覚への反応のほうが、欲求に直結している',       reverse: false },
  { id: 14, axis: 'pe', text: '気持ちが伴わない相手との行為は、どれだけ良くても虚しさが残る',   reverse: true  },
  { id: 15, axis: 'pe', text: '好みの外見でも、感情が動かなければ欲求はほとんど生まれない',     reverse: true  },

  // ── 偏食(C) ⇄ 雑食(W) 軸 ── high=C（こだわり強）, reverse=true → W（雑食）寄り ──
  { id: 16, axis: 'cw', text: '興奮するシチュや条件が細かすぎて、人に説明すると引かれることがある',         reverse: false },
  { id: 17, axis: 'cw', text: '理想の展開が頭の中にあって、そこからズレると一気に冷める',                   reverse: false },
  { id: 18, axis: 'cw', text: '刺さるシチュが絞られすぎていて、「また同じやつだ」と自分でも思う',           reverse: false },
  { id: 19, axis: 'cw', text: 'シチュや条件より雰囲気さえ合えば全然楽しめる、こだわりがほぼない',           reverse: true  },
  { id: 20, axis: 'cw', text: '「何でもいい」が本音で、特定の設定にこだわる感覚がよくわからない',           reverse: true  },
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

// 5段階回答(1〜5)から各軸スコアを計算
export function calcResult(answers: Record<number, number>): QuizResult {
  const axes: Axis[] = ['ds', 'nx', 'pe', 'cw'];
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

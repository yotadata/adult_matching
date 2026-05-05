export type Axis = 'ds' | 'nx' | 'pe' | 'hl';

export type QuizTypeKey =
  | 'snph' | 'snpl' | 'sneh' | 'snel'
  | 'sxph' | 'sxpl' | 'sxeh' | 'sxel'
  | 'mnph' | 'mnpl' | 'mneh' | 'mnel'
  | 'mxph' | 'mxpl' | 'mxeh' | 'mxel';

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
  { id: 1,  axis: 'ds', text: 'AV観るとき、自然とリードしてる側の目線になってる',                             reverse: false },
  { id: 2,  axis: 'ds', text: '相手の反応をコントロールしてる感覚が、一番の興奮源になってる',               reverse: false },
  { id: 3,  axis: 'ds', text: '「もっとして」って言わせたい側か言いたい側か、と問われたら迷わず言わせたい側', reverse: false },
  { id: 4,  axis: 'ds', text: '相手に好き勝手にされると、むしろ気持ちよくなってしまう',                     reverse: true  },
  { id: 5,  axis: 'ds', text: '尽くしすぎて後で我に返ることが、定期的にある',                               reverse: true  },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── high=N, reverse=true → X寄り ──
  { id: 6,  axis: 'nx', text: 'コスプレや設定より、普段着のままの流れの方が逆に興奮する',                   reverse: false },
  { id: 7,  axis: 'nx', text: 'スーパーや職場で偶然二人になった流れから、という妄想がふと来ることがある',   reverse: false },
  { id: 8,  axis: 'nx', text: 'エロい漫画や動画より、現実にありそうなリアルな設定の方が断然刺さる',         reverse: false },
  { id: 9,  axis: 'nx', text: '制服や特殊な設定を入れると、目の前の人が別の存在になる感覚がある',           reverse: true  },
  { id: 10, axis: 'nx', text: '現実にはあり得ない設定・世界観の方が、欲求の最大要因になってる',             reverse: true  },

  // ── 快楽(P) ⇄ 感情(E) 軸 ── high=P, reverse=true → E寄り ──
  { id: 11, axis: 'pe', text: 'タイプの外見の人に会って3秒で「あ、来た」ってなることがある',               reverse: false },
  { id: 12, axis: 'pe', text: '感情とか関係なく、身体的に合う相手ならそれなりに満足できる',                 reverse: false },
  { id: 13, axis: 'pe', text: '妄想するとき、顔や感情より身体のパーツや動きに焦点が当たりがち',             reverse: false },
  { id: 14, axis: 'pe', text: 'どれだけ外見がタイプでも、気持ちが動かないと途中で冷めてしまう',             reverse: true  },
  { id: 15, axis: 'pe', text: '感情のつながりがないと、終わった後にじわじわ虚しくなる',                     reverse: true  },

  // ── 偏食(H) ⇄ 雑食(L) 軸 ── high=H（こだわり強）, reverse=true → L（雑食）寄り ──
  { id: 16, axis: 'hl', text: '興奮するシチュや条件が細かすぎて、人に説明すると引かれることがある',         reverse: false },
  { id: 17, axis: 'hl', text: '理想の展開が頭の中にあって、そこからズレると一気に冷める',                   reverse: false },
  { id: 18, axis: 'hl', text: '刺さるシチュが絞られすぎていて、「また同じやつだ」と自分でも思う',           reverse: false },
  { id: 19, axis: 'hl', text: 'シチュや条件より雰囲気さえ合えば全然楽しめる、こだわりがほぼない',           reverse: true  },
  { id: 20, axis: 'hl', text: '「何でもいい」が本音で、特定の設定にこだわる感覚がよくわからない',           reverse: true  },
];

export interface AxisScore {
  pct: number;     // 0〜100（高いほどD/N/P/H寄り）
  label: string;   // 「ガチ偏食」「偏食寄り」など
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

  const d = scores.ds.isHigh ? 's' : 'm';
  const n = scores.nx.isHigh ? 'n' : 'x';
  const p = scores.pe.isHigh ? 'p' : 'e';
  const h = scores.hl.isHigh ? 'h' : 'l';
  const typeKey = `${d}${n}${p}${h}` as QuizTypeKey;

  return { typeKey, scores };
}

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  snph: {
    key: 'snph', name: '肉食系番長', emoji: '🔥',
    tagline: '声出させることに、異常な達成感がある',
    description: '相手の反応をコントロールするのが一番の興奮源で、リアクションが薄いと正直テンションが下がる。日常のなんでもない瞬間に欲求スイッチが入るので、相手からすると突然感がある。興奮トリガーが細かすぎて、設定が少し違うだけで急に萎えるという繊細な一面もあり、「なんで今冷めたの」と言われたことが一度はある。',
    color: '#FF6B6B', accent: '#C0392B',
  },
  snpl: {
    key: 'snpl', name: '静かに仕留める職人', emoji: '🧊',
    tagline: '急がない、でも気づいたら仕留めてる',
    description: 'リードするのが自然体で、気合いを入れてやっているわけじゃない。日常のちょっとした場面でさらっと動けて、相手や状況へのこだわりが特にない。「なんでもいい、いい感じになれば」が本音で、そのラフさが逆に相手を安心させている。仕留めるときの淡々とした確実さが、いつの間にか相手をハマらせる。',
    color: '#4ECDC4', accent: '#1A9B8C',
  },
  sneh: {
    key: 'sneh', name: '感情直結型リーダー', emoji: '❤️‍🔥',
    tagline: '気持ちが動いた瞬間、欲求もセットで来る',
    description: '好きな相手に対しては日常のあらゆる場面で欲求が発動するので、自分でも「また来た」と思うことがある。リードしながらも感情が顔に出るため「何考えてるか全部わかる」とよく言われる。興奮の条件がかなり細かく「この雰囲気じゃないと違う」と感じると急に引く。「なんで今冷めたの」は人生で何度も聞いたセリフ。',
    color: '#FF8E53', accent: '#E74C3C',
  },
  snel: {
    key: 'snel', name: 'じわじわくるカリスマ', emoji: '🌙',
    tagline: 'あまり語らないのに、気づいたら支配されてる',
    description: '感情が動けば欲求も来るが、特定のシチュへのこだわりはほぼない。「どこでも、相手次第」が本音で、この柔軟さが場の安心感を作っている。静かなのに存在感があって、気づいたら相手が夢中になっているのは本人も少し不思議に思っている。「怖くない？」と聞かれることが定期的にある。',
    color: '#A29BFE', accent: '#6C5CE7',
  },
  sxph: {
    key: 'sxph', name: '設定厨の征服者', emoji: '⚡',
    tagline: '設定が合わないと、急に萎えるという繊細さがある',
    description: 'コスプレや特殊な設定が入ると目の前の人が別の存在になる感覚があって、そこからが本番。「こんなことしてみたい」リストが常に更新されていて、欲求のトリガーが特殊すぎる。設定が少しでもズレると急に冷めるので、自分でも「細かいな」と思うことがある。一般向けに説明するとたいてい「え」となる。',
    color: '#FD79A8', accent: '#E84393',
  },
  sxpl: {
    key: 'sxpl', name: 'ゆるい一匹狼', emoji: '🐺',
    tagline: '非日常感さえあれば、こだわりは特にない',
    description: '非日常な演出があれば、設定の種類にはこだわらない。コスプレでも役割設定でも「非日常感」さえあれば全部乗れる。リードしながらどんな設定にも柔軟に対応できるので、相手のやりたいことを叶えやすい。ミステリアスな雰囲気があるが、本人は至って気楽にやっている。',
    color: '#636E72', accent: '#2D3436',
  },
  sxeh: {
    key: 'sxeh', name: '劇場型支配者', emoji: '🎭',
    tagline: 'あのシナリオじゃないと来ない、が多い',
    description: '特定のシナリオで感情に火がついたとき、スイッチが入る。「このシチュでこう来る」という細かいこだわりがあって、そこからズレると急に冷める。条件が揃わないと本気にならないが、揃ったときの演技力と熱量は本物。「なんでその設定にこだわるの？」と聞かれても、うまく説明できない。',
    color: '#FDCB6E', accent: '#E17055',
  },
  sxel: {
    key: 'sxel', name: '感情レア発火型', emoji: '🌌',
    tagline: 'めったに来ないが、来たときは本物',
    description: '非日常な状況で感情が動けば、設定の細かさは問わない。動くときは全力で、動かないときは全然動かない。感情トリガーはシンプルで、設定へのこだわりよりその場の感情の揺れが重要。「本気出したら怖い」と思われているらしいが、本人はそんなに本気を出す機会がない。',
    color: '#6C5CE7', accent: '#4A3AB5',
  },
  mnph: {
    key: 'mnph', name: '反応コレクター', emoji: '🌿',
    tagline: '相手の反応を全種類引き出したいタイプ',
    description: '相手の声・表情・身体の反応を「もっと出したい」という欲求が強くて、気づくとやりすぎている。日常の流れで動くタイプだが、欲求の発動条件が細かく「このシチュじゃないと来ない」が結構多い。「尽くしすぎ」と言われることと、設定が合わないと急に萎えることが、同じ人の中に共存している。',
    color: '#55EFC4', accent: '#00B894',
  },
  mnpl: {
    key: 'mnpl', name: 'なんでもOK世話焼き', emoji: '🌸',
    tagline: '相手が気持ちよければ、こだわりは特にない',
    description: '相手が気持ちよければそれでいい、というシンプルな動機。日常の延長でさらっと尽くせて、シチュや条件へのこだわりが特にない。「何でもいい、相手次第」が本音で、この柔軟さが相手を安心させている。尽くすことが自然体なので、気づいたら相手はかなり満足している。',
    color: '#81ECEC', accent: '#00CEC9',
  },
  mneh: {
    key: 'mneh', name: '溺愛スイッチ持ち', emoji: '💝',
    tagline: '好きだと、欲求が止まらなくなるやつ',
    description: '好きな相手への欲求頻度が高く、日常のどんな場面でも来る。相手の感情状態に敏感で、「なんか元気なさそう」でスイッチが入ることもある。尽くしたい欲と「この流れじゃないと違う」というこだわりが同居していて、条件が合わないと急に気持ちが引く。「さっきまで積極的だったのに」となることが定期的にある。',
    color: '#FF7675', accent: '#D63031',
  },
  mnel: {
    key: 'mnel', name: '静かな献身家', emoji: '🕊️',
    tagline: '心がつながれば、設定は何でもいい',
    description: '感情が動くことがすべてのトリガーで、シチュへのこだわりはほぼない。日常の延長で相手の感情に寄り添うことに充実感がある。「また会いたい」と思われる深さがあるが、本人はそんなに意識していない。派手さはないのに、じわじわ存在感が大きくなるタイプ。',
    color: '#DFE6E9', accent: '#B2BEC3',
  },
  mxph: {
    key: 'mxph', name: '妄想持ち込みサービス型', emoji: '🍬',
    tagline: '頭の中の設定と、実際の行動のギャップが激しい',
    description: '頭の中は非日常的な設定の妄想でいっぱいだが、実際は相手の快楽を引き出すことに全力で動く。この落差がすごい。妄想の設定が細かすぎて「実はこういうシチュを想定してた」と打ち明けたら「え、そこまで考えてたの」となった経験がある。非日常な演出があると欲求が跳ね上がるが、条件がズレると「違う」となる。',
    color: '#FAB1A0', accent: '#E17055',
  },
  mxpl: {
    key: 'mxpl', name: 'なんでも乗れる夢想家', emoji: '✨',
    tagline: 'どんな設定にも全力で乗れる守備範囲の広さ',
    description: 'コスプレでも役割設定でも、非日常な雰囲気さえあればなんでも全力で乗れる。特定の設定へのこだわりより柔軟さが特徴で、相手の快楽を引き出しながらどんな設定にも楽しく対応できる。「どんな設定でも大丈夫」という守備範囲の広さが密かな自慢で、断ったことがほぼない。',
    color: '#FFF4E6', accent: '#FDCB6E',
  },
  mxeh: {
    key: 'mxeh', name: 'シナリオ解放型感情家', emoji: '💫',
    tagline: '条件が揃ったら、誰より燃えるやつ',
    description: '特定のシナリオで感情に火がつく構造になっていて、条件が揃わないと本気にならない。「その設定じゃないとダメなの？」と聞かれると「そう」としか言えない。揃ったときの熱量は誰より高く、手がつけられなくなる。普段はそんなに見えないのに、スイッチが入ると周囲がびっくりする。',
    color: '#FD79A8', accent: '#E84393',
  },
  mxel: {
    key: 'mxel', name: 'レアスイッチの月の人', emoji: '🌕',
    tagline: '来たら全力、来なければ無',
    description: '非日常な状況で感情が動けば、設定の細かさは問わない。心が動いたときにだけスイッチが入る、シンプルな構造。めったに来ないが、来たときは全力で深い。感情が揺れさえすれば何でも乗れるという柔軟さが、来たときの爆発力を支えている。「急に本気になるの怖い」と言われたことが一度はある。',
    color: '#C7ECEE', accent: '#74B9FF',
  },
};

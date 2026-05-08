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
    labelHigh: '頻度高',
    labelLow: '頻度低',
    degreesHigh: ['ド頻度高', '頻度高め'],
    degreesLow: ['ド頻度低', '頻度低め'],
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

  // ── 頻度高(H) ⇄ 頻度低(L) 軸 ── high=H, reverse=true → L寄り ──
  { id: 16, axis: 'hl', text: '性的な欲求は頻繁にある方だと思う',                               reverse: false },
  { id: 17, axis: 'hl', text: '間を空けるより、できるだけ多く楽しみたい',                       reverse: false },
  { id: 18, axis: 'hl', text: '欲求が来たらなかなか抑えられない',                               reverse: false },
  { id: 19, axis: 'hl', text: '回数より、一度の深さや濃さを大切にしたい',                       reverse: true  },
  { id: 20, axis: 'hl', text: '性的な行為は特別な機会として、じっくり楽しみたい',               reverse: true  },
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

  const d = scores.ds.isHigh ? 's' : 'm';
  const n = scores.nx.isHigh ? 'n' : 'x';
  const p = scores.pe.isHigh ? 'p' : 'e';
  const h = scores.hl.isHigh ? 'h' : 'l';
  const typeKey = `${d}${n}${p}${h}` as QuizTypeKey;

  return { typeKey, scores };
}

export const QUIZ_TYPES: Record<QuizTypeKey, QuizType> = {
  snph: {
    key: 'snph', name: '大悪狼', emoji: '🐺',
    tagline: '追い込むほど燃える。止まれない捕食者',
    description: '衝動が来たら止まらない。赤ずきんを追い詰める狼のように、相手より先に動いて流れを自分で作る。「もっと」が口癖で、物足りなさを感じると自分からアクションを起こす。リードする側にいると本領発揮するタイプ。',
    color: '#C0392B', accent: '#922B21',
  },
  snpl: {
    key: 'snpl', name: '過食の魔女', emoji: '🍖',
    tagline: '急がない。でも絶対に食べる',
    description: 'ヘンゼルを太らせた魔女のように、じっくりペースを作って確実に相手を手中に収める。焦らない。でも決めたらやり遂げる。日常の中でさりげなく主導権を握るのが得意なタイプ。',
    color: '#1A9B8C', accent: '#148070',
  },
  sneh: {
    key: 'sneh', name: '呪われた野獣', emoji: '🌹',
    tagline: '気持ちに火がついたら、誰も止められない',
    description: '感情と欲求がリンクしている。野獣のように、気持ちが盛り上がると行動も加速する。日常の積み重ねで興奮するタイプで、「好き」と「もっと」が同時に来る。リードしながらも感情豊か。',
    color: '#E67E22', accent: '#CA6F1E',
  },
  snel: {
    key: 'snel', name: '糸紡ぎの魔女', emoji: '🪄',
    tagline: '静かな呪いが、一番深く刺さる',
    description: '多くを語らず、でも確実に場の空気を掌握している。眠れる森の魔女のように、感情的になることは少ないが、その静けさ自体が圧になる。日常のリアルな場面で、じわじわと相手を引き込む。',
    color: '#7D3C98', accent: '#6C3483',
  },
  sxph: {
    key: 'sxph', name: '千夜の王', emoji: '👑',
    tagline: '毎夜違う刺激を、飽きるまで求める',
    description: '千夜一夜の王のように、日常の刺激では満足できない。特殊なシチュエーションや非日常的な体験に強く反応する。身体的な刺激への感度が高く、「こんなことしてみたい」リストが尽きない。',
    color: '#2E86C1', accent: '#1A5276',
  },
  sxpl: {
    key: 'sxpl', name: '本性隠しの蛙', emoji: '🐸',
    tagline: '滅多に化けない。でも化けたら本物',
    description: 'カエルの王子のように、欲求はあるが頻繁には動かない。独自のファンタジーや世界観を持っていて、日常にはない特別な状況にしか本気で反応しない。その落差が誰より大きい。',
    color: '#1E8449', accent: '#196F3D',
  },
  sxeh: {
    key: 'sxeh', name: '青ひげ男爵', emoji: '🗝️',
    tagline: '禁断の部屋に、感情のすべてを詰めた',
    description: '非日常的なシチュエーションに感情が乗ると、スイッチが入る。頻度よりも「盛り上がり」を重視。特定の設定やシナリオで感情が爆発する、感情トリガー型の欲求持ち。',
    color: '#2471A3', accent: '#1A5276',
  },
  sxel: {
    key: 'sxel', name: '海の魔女', emoji: '🐚',
    tagline: '滅多に動かない。でも動いたら全部持っていく',
    description: '人魚姫と契約を結ぶ海の魔女のように、めったに動かないが動くときは全力。非日常的な感情体験を重視していて、その瞬間の「深さ」がすべて。ファンタジーな設定で感情が動くと、誰より激しくなる。',
    color: '#154360', accent: '#0E2F44',
  },
  mnph: {
    key: 'mnph', name: '泡になる人魚', emoji: '🫧',
    tagline: '声を失っても、尽くすことをやめない',
    description: 'アンデルセンの人魚姫のように、相手の反応を見るのが何より好き。日常的な場面で、相手の身体的な満足を引き出すことに強い充実感を覚える。欲求の頻度も高く、尽くす回数も多い。',
    color: '#17A589', accent: '#148F77',
  },
  mnpl: {
    key: 'mnpl', name: '眠れる美女', emoji: '💤',
    tagline: '起こされたとき、全力で応える',
    description: '欲求もサービス精神も、波がある。眠れる森の美女のように、呼ばれたときだけ目が覚める。でもやると決めたら手加減しない。日常の中で相手の身体的な反応を引き出すことに喜びを感じる。',
    color: '#E91E8C', accent: '#C0166F',
  },
  mneh: {
    key: 'mneh', name: '赤い靴の娘', emoji: '🩰',
    tagline: '一度動き出したら、止まれない',
    description: 'アンデルセンの赤い靴のように、気持ちが動くと欲求も止まらなくなる。心のつながりを感じると火がつき、日常の中で頻繁に感情が揺れ動く。踊り続けずにいられない正直な人。',
    color: '#3498DB', accent: '#2874A6',
  },
  mnel: {
    key: 'mnel', name: '塔のラプンツェル', emoji: '🏰',
    tagline: '来てくれた相手に、全部渡す',
    description: '心がつながったと感じるときにだけ、スイッチが入る。塔の中で待ち続けたラプンツェルのように、頻度より質派で、相手の感情に寄り添うことに快感がある。派手な刺激より、静かな充足を好む。',
    color: '#95A5A6', accent: '#717D7E',
  },
  mxph: {
    key: 'mxph', name: '毎夜語るシェヘラザード', emoji: '🌙',
    tagline: '毎晩新しい物語で、相手を引き込み続ける',
    description: '千一夜物語のシェヘラザードのように、頭の中に非日常的な設定が尽きない。相手の身体的な満足を引き出すことに全力で、頻繁に動く。続きが気になると言わせ続けるタイプ。',
    color: '#E59866', accent: '#CA6F1E',
  },
  mxpl: {
    key: 'mxpl', name: 'お菓子の家のグレーテル', emoji: '🍬',
    tagline: '理想の場面へのこだわりが、人並み外れている',
    description: 'グレーテルのように、普通の場面では動かない。理想のシチュエーションや非現実的な設定にだけ、身体が正直に反応する。頻度は少ないが、その分「完璧な場面」へのこだわりが強く、妄想の解像度が異常に高い。',
    color: '#E74C3C', accent: '#CB4335',
  },
  mxeh: {
    key: 'mxeh', name: '月へ帰るかぐや', emoji: '🎋',
    tagline: '感情が動いたら、月まで行ってしまう',
    description: '非日常的な状況や特殊な関係性の中で感情が動いたとき、欲求が最大化する。かぐや姫のように、心が揺れるほど欲しくなる感情連動型。頻度も高く、感情的な盛り上がりを何度も求めてしまう。',
    color: '#9B59B6', accent: '#884EA0',
  },
  mxel: {
    key: 'mxel', name: '花の中の親指姫', emoji: '🌸',
    tagline: '人生に数回、本当に動くときがある',
    description: 'めったに欲求が来ないが、来たときは深い。親指姫のように、非日常的な感情体験との組み合わせでしか本気にならない。心が完全に動いた相手には全力で応じる、一生に数回型の本気タイプ。',
    color: '#5D6D7E', accent: '#4D5D6E',
  },
};

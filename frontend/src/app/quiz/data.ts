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
  detailDescription: string;
  favPlay: string[];
  trivia: string;
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
  { id: 4,  axis: 'pe', optionA: '顔とスタイルが好みなら、性格や関係性は関係なく惹かれる', optionB: 'どれだけ好みでも、気持ちが入らない相手とは無理' },
  { id: 5,  axis: 'pe', optionA: '気持ちよければ相手への感情は関係ない',                  optionB: '好きじゃない相手だと気持ちよくても虚しい' },
  { id: 6,  axis: 'pe', optionA: '会って数分で「この人とやりたい」と思うことがある',       optionB: 'ある程度仲良くなった相手じゃないと欲求が生まれない' },

  // ── 日常(N) ⇄ 非日常(X) 軸 ── A=N側, B=X側 ──
  { id: 7,  axis: 'nx', optionA: '毎日会う相手でも欲求は普通に続く',                    optionB: '旅行先やホテルなど、いつもと違う場所だと明らかに興奮が上がる' },
  { id: 8,  axis: 'nx', optionA: '慣れ親しんだ相手の方が安心して本気になれる',          optionB: '初対面に近い緊張感のある相手の方が燃える' },
  { id: 9,  axis: 'nx', optionA: 'いつもの部屋・いつもの雰囲気で十分テンションが上がる', optionB: '「バレるかも」という状況や、非日常の緊張感がないと物足りない' },

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
    detailDescription: '相手の反応を観察しながら攻め方を変えていくのが好きで、弱い部分を見つけた瞬間から集中的に攻め続ける。「こういう反応が見たい」という明確なビジョンがあり、それが達成されるまでやめない。タイプ外の相手にはそもそも欲求が湧かない。',
    favPlay: ['言葉責め', '焦らし・じらし', '目隠し', '体の反応を観察するプレイ', '相手が限界になるまで攻め続ける'],
    trivia: '「タイプじゃないと無理」と言いながら、タイプの相手には異常に細かいところまで覚えてる。',
    color: '#FF6B6B', accent: '#C0392B',
  },
  spnw: {
    key: 'spnw', name: '全方位ハンター', emoji: '🧊',
    tagline: '誰でも落とせる、どこでも本気',
    description: '外見や身体の相性で動き、リードするのが好き。慣れた相手でも新しい相手でも十分燃えるし、細かいこだわりもない。「まあなんでもいけるし、リードさせてくれれば」という間口の広さが武器。',
    detailDescription: '相手の好みを素早く読み取って合わせながら、自分がリードする形に自然に持っていくのが得意。こだわりが少ない分、どんな相手でも楽しめてしまう。「これが好き？じゃあこれは？」と試しながら攻めるスタイル。',
    favPlay: ['リードしながらの前戯', '体位や展開を自分で決める', '相手の好みを引き出すプレイ', '主導権を握ったままのセックス'],
    trivia: '「誰とでもいける」と思われがちだけど、実はリードさせてもらえない相手には全然燃えない。',
    color: '#4ECDC4', accent: '#1A9B8C',
  },
  senc: {
    key: 'senc', name: 'むっつり帝王', emoji: '❤️‍🔥',
    tagline: '感情のスイッチが入ると、誰より熱い',
    description: '感情が動いた相手をリードしたくなる。慣れ親しんだ日常の中でもしっかり燃えるが、感情のトリガーが細かい。「この人のこういうところ」が揃わないと本気になれず、条件が揃った瞬間は誰より熱くなる。',
    detailDescription: '好きな相手への独占欲が強く出るタイプで、「この人を自分だけのものにしたい」という感情がプレイに直結する。普段は冷静に見えるのに、スイッチが入ると豹変する落差が特徴。相手の「あなただからこうなっちゃう」という反応に異常に弱い。',
    favPlay: ['独占欲を刺激するプレイ', '「好きだから」が伝わるセックス', '囁き・耳元での言葉責め', '相手が自分に溺れていく過程を楽しむ'],
    trivia: '普段はクールなのに、好きな人の前だと自分でも引くくらい独占欲が出てくる。',
    color: '#FF8E53', accent: '#E74C3C',
  },
  senw: {
    key: 'senw', name: '天然の王様', emoji: '🌙',
    tagline: '気持ちが乗れば、どこでも自然にリードする',
    description: '感情が動けばどんな状況でもリードできる。慣れた相手・場所でも十分OKで、細かいこだわりもない。その場の雰囲気を読んで自然にリードし、気づいたら相手が夢中になっている。一番「普通に最強」なタイプ。',
    detailDescription: '特別なセッティングや条件がなくても、好きな相手と気持ちが乗っていれば自然にリードできる。テクニックというより雰囲気で相手を引き込むタイプで、終わった後に「なんかすごかった」と言われやすい。',
    favPlay: ['雰囲気重視のセックス', 'その場の流れに乗ったプレイ', '相手の感情を高めながらリードする', '終始リラックスできる体位・展開'],
    trivia: '特に何も考えてないのに相手が夢中になってしまう。本人はなぜか分かっていない。',
    color: '#A29BFE', accent: '#6C5CE7',
  },
  spxc: {
    key: 'spxc', name: '刺激ジャンキー', emoji: '⚡',
    tagline: '非日常の緊張感の中でしか、本気にならない',
    description: 'ホテルや旅先、バレるかもしれない場所など、いつもと違う刺激がないと物足りない。外見重視でリードしたいが、こだわりが細かく設定がズレると冷める。「いつもと違う場所で、自分のペースで支配する」が理想形。',
    detailDescription: '日常の延長では欲求が湧きにくく、「ここでしかできない」という緊張感があって初めてスイッチが入る。バレるかもしれない場所・高級ホテル・車内など、非日常の設定とタイプ外の相手という二重の縛りがある。揃ったときの集中力と興奮は異常。',
    favPlay: ['ホテルや旅先での開放感プレイ', '外での際どいシチュエーション', '非日常感を演出したプレイ', '「バレるかも」という緊張感の中で'],
    trivia: '旅行に行くとテンションが別人になる。普段が嘘みたいにアクティブになる。',
    color: '#FD79A8', accent: '#E84393',
  },
  spxw: {
    key: 'spxw', name: 'アドレナリン番長', emoji: '🎲',
    tagline: '非日常さえあれば、なんでもいける',
    description: 'いつもの場所では物足りない。ホテルでも旅先でも、非日常の空気さえあればこだわりなく全力になれる。外見重視でリードするのが好き。「普通の状況はつまらないが、どこでも遊べる」という自由人な支配タイプ。',
    detailDescription: '場所が変わるだけで別人のように積極的になる。相手の細かい条件よりも「いつもと違う」という空気の方が重要で、非日常さえあればリードしながらなんでも試せてしまう。旅先での一夜が最高の思い出になりやすい。',
    favPlay: ['旅先・ホテルでのフルスロットルプレイ', '普段はしないことを試す', 'リードしながら相手の限界を探る', '開放的な空間でのセックス'],
    trivia: '「旅行行こう」と言われると、自動的にテンションが上がる条件反射がある。',
    color: '#636E72', accent: '#2D3436',
  },
  sexc: {
    key: 'sexc', name: '過激な演出家', emoji: '🎭',
    tagline: '感情も場所も、全部揃って初めて動く',
    description: '感情が動いた相手をリードしたいが、非日常の場でないとスイッチが入らない。さらに感情のトリガーも細かい。条件が二重に絞られているぶん、全部揃ったときの爆発力はすさまじい。「なんか違う」が多い代わりに、刺さったときは全力。',
    detailDescription: '好きな相手との特別な場所・特別な状況でなければ本気になれない。条件が二重に絞られているぶん、普段は「なんか違う」と感じることが多い。ただし全部揃ったときの没入感と情熱は他のタイプの追随を許さない。',
    favPlay: ['旅先・ホテルでの感情爆発プレイ', 'ロールプレイや演出を加えたセックス', '特別感を演出した前戯', '「今夜だけ」感のあるシチュエーション'],
    trivia: '普段は淡白に見られがちなのに、条件が揃うと自分でも驚くくらい激しくなる。',
    color: '#FDCB6E', accent: '#E17055',
  },
  sexw: {
    key: 'sexw', name: '刺激の旅人', emoji: '🌌',
    tagline: '感情さえ動けば、非日常の中でどこでもリードする',
    description: '感情が動いた相手には強くリードしたくなる。ただ、いつもと同じ場所では物足りない。ホテルや旅先など非日常の空気の中で感情に火がつくと、こだわりなく全力になれる。感情主導で刺激を求めるロマンチストな支配者。',
    detailDescription: '好きな人と非日常の場所に行くことで感情と興奮が同時に爆発する。旅行や遠出がそのままロマンチックな体験につながりやすく、「あの旅のあの夜」みたいな特別な記憶を作るのが得意なタイプ。',
    favPlay: ['旅先での感情と興奮が重なる体験', '非日常の空気の中でリードする', '感情が高まってからの積極的なプレイ', 'ロマンチックな演出からの展開'],
    trivia: '旅行中に急に積極的になって、相手を驚かせた経験が一度はある。',
    color: '#6C5CE7', accent: '#4A3AB5',
  },
  mpnc: {
    key: 'mpnc', name: 'こだわり沼らせ魔', emoji: '🌿',
    tagline: '条件が揃った相手にだけ、全力で尽くす',
    description: '相手を喜ばせることが好きで、外見や身体の相性で相手を選ぶ。慣れた環境でも十分燃えるが、こだわりの条件が細かい。「この人だ」と思えた相手への集中力は異常で、気づいたら相手が深みにはまっている。',
    detailDescription: 'タイプの相手と判断したら、その相手が気持ちよくなることだけを考えて動く。何が好きか・どこが弱いかを徹底的にリサーチして覚えていて、それをすべて使ってくる。気づいたら相手が「この人以外じゃ満足できない」状態になっている。',
    favPlay: ['相手の弱点を覚えて使うプレイ', '徹底的な前戯', '相手の反応を引き出すことへの集中', 'リピートするほど精度が上がる尽くし'],
    trivia: '「そんなところ覚えてたの？」と言われることが定期的にある。',
    color: '#55EFC4', accent: '#00B894',
  },
  mpnw: {
    key: 'mpnw', name: '世話焼き八方美人', emoji: '🌸',
    tagline: '誰に対しても、どこでも尽くせる',
    description: '相手の身体的な満足を引き出すのが好きで、こだわりは少ない。慣れた相手でも新しい相手でも自分なりに尽くせる。「まあ誰でもいけるし、喜んでくれれば嬉しい」という、間口の広い頼もしい奉仕タイプ。',
    detailDescription: '相手が気持ちよさそうにしているのを見るのが一番の快感で、自分の欲求よりも相手の満足を優先する。特定の好みに縛られず、相手に合わせて対応を変えられるため、どんな相手にも高評価をもらえる。',
    favPlay: ['相手のペースに完全に合わせるプレイ', '長時間の前戯・後戯', '相手が求めることを先読みして動く', 'どんな体位・展開でも楽しめる'],
    trivia: '「あなたって器用だよね」と言われることが多いが、本人は特に意識していない。',
    color: '#F9E784', accent: '#F1C40F',
  },
  menc: {
    key: 'menc', name: 'むっつり溺愛体質', emoji: '💝',
    tagline: 'この感情、この人だけに全部渡す',
    description: '心がつながった相手に尽くしたくなるが、感情のトリガーが細かい。「この人のこういうところが好き」が揃って初めてスイッチが入る。慣れ親しんだ関係の中で愛情が深まるタイプで、刺さると溺愛モードに入る。',
    detailDescription: '普段は控えめなのに、好きな相手に対してだけ異常に尽くしたくなる。「この人を喜ばせたい」という感情が直接プレイに結びついていて、感情が動けば動くほど奉仕の精度が上がる。相手に「あなただから」と言わせることが最大の報酬。',
    favPlay: ['「好きだから尽くす」が伝わるプレイ', '相手が求める前に先回りして動く', '感情が高まってからの深い前戯', 'ゆっくり時間をかけた濃密なセックス'],
    trivia: '好きじゃない人には全くスイッチが入らないのに、好きになった瞬間に別人になる。',
    color: '#E84393', accent: '#C0136F',
  },
  menw: {
    key: 'menw', name: '一途な忠犬', emoji: '🕊️',
    tagline: '気持ちさえ動けば、どんな相手も受け止める',
    description: '心がつながったと感じれば、どんな状況でも相手に寄り添える。慣れた日常の中でじっくり愛情が育つタイプで、こだわりなく相手に合わせられる。静かで包容力があり、「あなたに合わせます」という姿勢が自然体。',
    detailDescription: '気持ちが動いた相手には無条件に寄り添おうとする。特定のプレイへのこだわりより「この人が求めるなら」という感情が優先されるため、相手にとって理想の奉仕をしてしまう。後から「あんなこともしてたな」と自分で驚くことがある。',
    favPlay: ['相手に完全に委ねるプレイ', '「なんでもいい」を本当に体現する', '感情がこもった抱擁・スキンシップ', '相手のリードに素直に応えるセックス'],
    trivia: '「好きじゃない人とは無理」が口癖なのに、好きになったら秒で沼る。',
    color: '#DFE6E9', accent: '#B2BEC3',
  },
  mpxc: {
    key: 'mpxc', name: '指示多めのしもべ', emoji: '🍬',
    tagline: '非日常の場で、決まった条件が揃って初めて動く',
    description: 'ホテルや旅先など非日常の環境で、外見が好みの相手に尽くすのが理想。こだわりの条件が細かく、設定がズレると冷める。「この場所で、この人に、こういうふうに尽くしたい」という妄想がクリアに決まっている。',
    detailDescription: '頭の中に「理想のシチュエーション」が明確に存在していて、それに近い状況でないと本気になれない。ホテルのベッドの上で、タイプの相手に、自分のペースで尽くすという設定が揃ったときの没入感は格別。指示されながら尽くすのが特に好き。',
    favPlay: ['相手に指示されながら尽くすプレイ', 'ホテルや旅先での奉仕', '設定・シチュエーションを決めてから始める', 'タイプの相手への徹底的な奉仕'],
    trivia: '妄想の解像度が異常に高くて、頭の中で完全なシナリオができあがっている。',
    color: '#FAB1A0', accent: '#E17055',
  },
  mpxw: {
    key: 'mpxw', name: 'スキンシップ魔人', emoji: '✨',
    tagline: 'いつもと違う場所さえあれば、なんでも尽くせる',
    description: '日常では物足りない。ホテルでも旅先でも、非日常の空気さえあれば相手の身体的な満足を引き出すことに全力になれる。細かいこだわりは薄く、「非日常さえあれば何でもいける」という柔軟な刺激追求型の尽くし屋。',
    detailDescription: '非日常の空気の中では自分でも驚くほど積極的に尽くせてしまう。相手への細かいこだわりより「場の空気」が重要で、ホテルのシャワー後・旅先の夜など、少し特別な空気があるだけで全力になれる。',
    favPlay: ['旅先・ホテルでのサービス全開プレイ', '開放的な空間での尽くし', 'お風呂・温泉など非日常の場での前戯', '相手の身体的満足を最優先にする'],
    trivia: '旅行中に「普段と全然違う」と言われることが多い。家だと謎におとなしい。',
    color: '#FFF4E6', accent: '#FDCB6E',
  },
  mexc: {
    key: 'mexc', name: 'こじらせの極み', emoji: '💫',
    tagline: '感情も場所も揃って、初めて全部渡せる',
    description: '心が動いた相手に尽くしたいが、非日常の場でないとスイッチが入らない。さらに感情のトリガーも細かい。条件が二重に絞られているぶん、全部揃ったときの没入度はすさまじい。旅先やホテルで感情が爆発する愛情家。',
    detailDescription: '普段の生活の中ではほとんどスイッチが入らないのに、好きな人と特別な場所に行くと別人のように全力になる。条件が揃うまでが長い分、解放されたときの溺愛エネルギーが凄まじく、相手が引くことがある。',
    favPlay: ['感情が爆発した後の全力奉仕', '旅先での「全部あげる」プレイ', '特別な夜の徹底的な尽くし', '好きな人と非日常の場所でする深いセックス'],
    trivia: '普段は淡白に見られていて、好きな人との旅行で初めて本性がバレる。',
    color: '#E17055', accent: '#C0392B',
  },
  mexw: {
    key: 'mexw', name: '夢見る妄想家', emoji: '🌕',
    tagline: '感情さえ動けば、どんな非日常でも全力で尽くす',
    description: '感情が動いた相手には全力で尽くしたくなる。いつもの場所より非日常の空気の中の方が燃える。細かいこだわりは薄く、ホテルでも旅先でも感情スイッチが入れば何でもOK。感情主導で動く、刺激を求める献身家。',
    detailDescription: '好きな人と非日常の場所に行くことで感情と奉仕欲が同時に爆発する。相手のこだわりより「好きな人と特別な夜」という状況そのものが大事で、そこでなら何でも尽くせてしまう。終わった後に「あんなことするとは思わなかった」と自分で驚く。',
    favPlay: ['好きな人との旅先での全力奉仕', '感情が先行してからの尽くし', '非日常の空気の中でやりたいことを全部やる', 'ロマンチックな展開からの深い奉仕'],
    trivia: '好きな人ができると急に旅行の計画を立て始める。本人はそれに気づいていない。',
    color: '#C7ECEE', accent: '#74B9FF',
  },
};

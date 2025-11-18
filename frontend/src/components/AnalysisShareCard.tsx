'use client';

import { forwardRef, useEffect, useMemo, useState } from 'react';
import QRCode from 'qrcode';
import type { AnalysisPerformer, AnalysisSummary, AnalysisTag } from '@/hooks/useAnalysisResults';

interface AnalysisShareCardProps {
  summary: AnalysisSummary | null;
  topTags: AnalysisTag[];
  topPerformers: AnalysisPerformer[];
  shareUrl: string;
}

type TypeProfile = {
  keywords: string[];
  typeName: string;
  highlight: string;
  keywordLabel: string;
  headerCopy: string;
};

const BACKGROUND_GRADIENT = 'linear-gradient(135deg, #d7d8ff 0%, #ffc8da 45%, #ffe6cf 100%)';

const TYPE_PROFILES: TypeProfile[] = [
  {
    keywords: ['美少女', 'ロリ', '制服', '女子校生', '妹'],
    typeName: '清楚ロリ系',
    highlight: '透明感あるかわいいムードでときめきがち。',
    keywordLabel: 'かわいい系',
    headerCopy: 'あどけない雰囲気の作品を追いかける王道ロリ派。',
  },
  {
    keywords: ['巨乳', '爆乳', 'お姉さん', 'むちむち', '痴女'],
    typeName: '巨乳お姉さん系',
    highlight: '包容力たっぷりの大人なお姉さんでとろけたい。',
    keywordLabel: 'ボリューム系',
    headerCopy: '余裕のあるお姉さんに甘やかされたい欲が爆発中。',
  },
  {
    keywords: ['人妻', '不倫', '寝取られ', 'NTR', '禁断'],
    typeName: '背徳NTR系',
    highlight: '禁断のストーリーで心拍を上げるのが快感。',
    keywordLabel: '背徳系',
    headerCopy: '日常では味わえない背徳展開に特大の刺さり方。',
  },
  {
    keywords: ['ギャル', '日焼け', 'ビッチ', 'パリピ'],
    typeName: 'ギラギラギャル系',
    highlight: '勢いとノリで押してくるギャルに弱い。',
    keywordLabel: 'ギャル系',
    headerCopy: '明るいギャルエネルギーでテンションを上げるタイプ。',
  },
  {
    keywords: ['単体作品', '王道', 'ノーマル'],
    typeName: '王道単体派',
    highlight: '企画よりもシンプルな単体作で集中したい。',
    keywordLabel: 'シンプル構成',
    headerCopy: 'じっくり堪能できる王道スタイルを求める派。',
  },
  {
    keywords: ['SM', '拘束', '調教', '支配', '主従'],
    typeName: '刺激スパイス系',
    highlight: 'ピリ辛な刺激で一気にスイッチが入る。',
    keywordLabel: 'スパイス系',
    headerCopy: '刺激的なシチュをアクセントにしたい攻め志向。',
  },
];

const EXTRA_KEYWORDS: Record<string, string> = {
  美少女: 'かわいい系',
  制服: '青春ムード',
  ロリ: 'あどけなさ',
  巨乳: 'ボリューム系',
  お姉さん: '大人ムード',
  痴女: '攻め気質',
  人妻: '大人っぽい',
  不倫: '背徳感',
  ギャル: 'ギラギラ系',
  単体作品: 'シンプル構成',
  企画: 'アイデア派',
  調教: 'スパイス',
  SM: '刺激系',
};

const FALLBACK_HIGHLIGHTS = [
  '作品選びはフィーリング重視。ハマると一気に沼るタイプ。',
  '雰囲気づくりと世界観が刺されば「気になる」確定。',
  'ストーリーよりも抜け感を優先する実用派。',
];

const DEFAULT_PROFILE = {
  typeName: 'バランス型',
  highlight: 'ジャンルを渡り歩きつつ、その日の気分で決める柔軟派。',
  keywordLabel: 'バランス志向',
  headerCopy: '王道からマニアックまで幅広くいける万能スタイル。',
};

const TAG_CATEGORY_MAP: Record<string, 'look' | 'body' | 'situation'> = {
  美少女: 'look',
  ロリ: 'look',
  制服: 'look',
  ギャル: 'look',
  人妻: 'situation',
  不倫: 'situation',
  寝取られ: 'situation',
  NTR: 'situation',
  単体作品: 'situation',
  調教: 'situation',
  中出し: 'body',
  巨乳: 'body',
  爆乳: 'body',
};

const TAG_CATEGORY_LABELS: Record<'look' | 'body' | 'situation', string> = {
  look: 'ルックス系',
  body: '体型系',
  situation: 'シチュ系',
};

const formatCount = (value: number | undefined | null) => {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('ja-JP');
};

const deriveTypeProfile = (tags: AnalysisTag[]) => {
  const tagNames = tags.map((tag) => tag.tag_name);
  const profile = TYPE_PROFILES.find((candidate) =>
    candidate.keywords.some((keyword) => tagNames.some((name) => name.includes(keyword))),
  );
  return profile ?? DEFAULT_PROFILE;
};

const mapTagToKeyword = (tagName: string) => {
  const profile = TYPE_PROFILES.find((candidate) =>
    candidate.keywords.some((keyword) => tagName.includes(keyword)),
  );
  if (profile) return profile.keywordLabel;
  const extra = Object.entries(EXTRA_KEYWORDS).find(([keyword]) => tagName.includes(keyword));
  return extra ? extra[1] : null;
};

const collectKeywordLabels = (typeProfile: TypeProfile | typeof DEFAULT_PROFILE, tags: AnalysisTag[]) => {
  const labels = new Set<string>();
  labels.add(typeProfile.keywordLabel);
  tags.forEach((tag) => {
    const mapped = mapTagToKeyword(tag.tag_name);
    if (mapped) labels.add(mapped);
  });
  const list = Array.from(labels).filter(Boolean);
  return list.length > 0 ? list.slice(0, 4) : ['バランス志向'];
};

const buildHighlights = (
  typeProfile: TypeProfile | typeof DEFAULT_PROFILE,
  topPerformers: AnalysisPerformer[],
  keywordLabels: string[],
) => {
  const highlightTexts: string[] = [];
  highlightTexts.push(typeProfile.highlight);

  if (topPerformers.length > 0) {
    const names = topPerformers.slice(0, 2).map((performer) => performer.performer_name).filter(Boolean);
    const performerText =
      names.length === 1
        ? `推し女優は ${names[0]}。出演すると即「気になる」を押す。`
        : `推し女優は ${names.join(' / ')}。この組み合わせが出ると優勝。`;
    highlightTexts.push(performerText);
  }

  if (keywordLabels.length > 0) {
    highlightTexts.push(`${keywordLabels[0]} のムードにハマりがち。`);
  }

  const fallbacks = [...FALLBACK_HIGHLIGHTS];
  while (highlightTexts.length < 3 && fallbacks.length > 0) {
    const fallback = fallbacks.shift();
    if (fallback) highlightTexts.push(fallback);
  }

  return highlightTexts.slice(0, 3);
};

const mapTagToCategory = (tagName: string): 'look' | 'body' | 'situation' => {
  const entry = Object.entries(TAG_CATEGORY_MAP).find(([keyword]) => tagName.includes(keyword));
  if (entry) return entry[1];
  return 'situation';
};

const buildTagEvidence = (tags: AnalysisTag[]) => {
  const totalLikes = tags.reduce((sum, tag) => sum + (tag.likes ?? 0), 0);
  const categoryCounts: Record<'look' | 'body' | 'situation', number> = {
    look: 0,
    body: 0,
    situation: 0,
  };

  tags.forEach((tag) => {
    categoryCounts[mapTagToCategory(tag.tag_name)] += tag.likes ?? 0;
  });

  const categoryEvidence = (Object.keys(categoryCounts) as Array<'look' | 'body' | 'situation'>).map((key) => ({
    label: TAG_CATEGORY_LABELS[key],
    ratio: totalLikes > 0 ? Math.round((categoryCounts[key] / totalLikes) * 100) : 0,
  }));

  const tagTop = tags
    .filter((tag) => tag.likes > 0)
    .sort((a, b) => (b.like_ratio ?? 0) - (a.like_ratio ?? 0))
    .slice(0, 3)
    .map((tag, index) => ({
      rank: index + 1,
      name: tag.tag_name,
      ratio: `${((tag.like_ratio ?? 0) * 100).toFixed(1)}%`,
      sample: `${tag.likes ?? 0}件`,
    }));

  const comment = 'ルックス系が中心で、上位タグも“かわいい系”に集中しています。';
  return { categoryEvidence, tagTop, comment };
};

const buildPerformerEvidence = (performers: AnalysisPerformer[]) => {
  const performerTop = performers
    .filter((performer) => performer.likes > 0)
    .sort((a, b) => (b.likes ?? 0) - (a.likes ?? 0))
    .slice(0, 3)
    .map((performer, index) => ({
      rank: index + 1,
      name: performer.performer_name,
      sample: `${performer.likes ?? 0}件`,
      share: `${((performer.share ?? 0) * 100).toFixed(1)}%`,
    }));

  const comment = '“かわいい系の女優”が気になる作品の大半を占めています。';
  return { performerTop, comment };
};

const AnalysisShareCard = forwardRef<HTMLDivElement, AnalysisShareCardProps>(function AnalysisShareCard(
  { summary, topTags, topPerformers, shareUrl },
  ref,
) {
  const [qrDataUrl, setQrDataUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!shareUrl) {
      setQrDataUrl(null);
      return () => {
        cancelled = true;
      };
    }
    QRCode.toDataURL(shareUrl, {
      width: 320,
      margin: 1,
      color: { dark: '#111827', light: '#ffffff' },
    })
      .then((url) => {
        if (!cancelled) setQrDataUrl(url);
      })
      .catch(() => {
        if (!cancelled) setQrDataUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [shareUrl]);

  const typeProfile = useMemo(() => deriveTypeProfile(topTags), [topTags]);
  const keywordLabels = useMemo(() => collectKeywordLabels(typeProfile, topTags), [typeProfile, topTags]);
  const highlightTexts = useMemo(
    () => buildHighlights(typeProfile, topPerformers, keywordLabels),
    [typeProfile, topPerformers, keywordLabels],
  );
  const featuredPerformers = useMemo(
    () =>
      topPerformers
        .slice(0, 2)
        .map((performer) => performer.performer_name)
        .filter(Boolean),
    [topPerformers],
  );
  const tagEvidence = useMemo(() => buildTagEvidence(topTags), [topTags]);
  const performerEvidence = useMemo(() => buildPerformerEvidence(topPerformers), [topPerformers]);

  const keywordPills = keywordLabels.slice(0, 4);
  const totalDecisions = formatCount(summary?.sample_size);

  return (
    <div
      ref={ref}
      id="share-card"
      className="ShareCardRoot"
      style={{
        position: 'relative',
        width: '1920px',
        height: '1080px',
        padding: '32px 40px',
        boxSizing: 'border-box',
        borderRadius: '32px',
        overflow: 'hidden',
        background: BACKGROUND_GRADIENT,
        fontFamily:
          '"Noto Sans JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <div
        className="ShareCardFrame"
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          borderRadius: '40px',
          boxSizing: 'border-box',
          background: 'rgba(255, 255, 255, 0.88)',
          border: '1px solid rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(18px)',
          boxShadow: '0 24px 80px rgba(15, 23, 42, 0.18)',
          overflow: 'hidden',
          padding: '64px 80px 32px 64px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div
          className="ShareCardInner flex flex-col h-full w-full"
          style={{ width: '100%', height: '100%', padding: 0, margin: 0, boxSizing: 'border-box' }}
        >
          <div className="TopSection flex flex-col gap-6">
          <section className="HeaderRow flex items-start justify-between gap-10" style={{ marginBottom: '48px' }}>
            <div className="HeaderTextBlock flex flex-col gap-4" style={{ maxWidth: '68%' }}>
              <div>
                <p className="text-sm font-semibold tracking-[0.35em] text-rose-400 uppercase">あなたのおかずタイプ</p>
                <h1 className="text-[54px] leading-[1.05] font-black text-gray-900 mt-1">{typeProfile.typeName}</h1>
              </div>
              <p className="text-lg text-gray-600 leading-relaxed">{typeProfile.headerCopy}</p>
              <div>
                <span className="inline-flex items-center text-xs font-semibold text-gray-700 bg-[rgba(15,23,42,0.05)] px-4 py-1.5 rounded-full">
                  判定数 {totalDecisions} 件
                </span>
              </div>
            </div>
            <div
              className="QRStamp flex flex-col items-center text-center gap-1.5"
              style={{
                width: '152px',
                height: '152px',
                padding: '18px',
                borderRadius: '32px',
                background: 'rgba(255, 255, 255, 0.92)',
                boxShadow: '0 12px 30px rgba(15,23,42,0.18)',
                flexShrink: 0,
                marginLeft: '32px',
                marginTop: '8px',
              }}
            >
              <p className="text-[10px] font-semibold tracking-[0.35em] text-rose-400 uppercase">診断</p>
              <div className="w-[130px] h-[130px] bg-white rounded-[26px] border border-gray-200 shadow-inner flex items-center justify-center p-2">
                {qrDataUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={qrDataUrl} alt="Seiheki Lab QR" className="w-full h-full object-contain" />
                ) : (
                  <span className="text-xs text-gray-400">QR準備中...</span>
                )}
              </div>
              <p className="text-[10px] text-gray-500">seihekilab.com</p>
            </div>
          </section>

          <section className="HighlightsRow flex gap-4 w-full flex-nowrap" style={{ marginBottom: '24px' }}>
            {highlightTexts.slice(0, 3).map((text, index) => (
              <div
                key={`highlight-${index}`}
                className="rounded-[24px] bg-white shadow-[0_12px_30px_rgba(15,23,42,0.15)] border border-white/70 px-5 py-5 flex flex-col gap-2 min-w-0"
                style={{
                  minHeight: '140px',
                  width: 'calc((100% - 32px) / 3)',
                  flex: '0 0 calc((100% - 32px) / 3)',
                  wordBreak: 'break-word',
                }}
              >
                <p className="text-xs font-semibold tracking-[0.3em] uppercase text-rose-400">
                  HIGHLIGHT {String(index + 1).padStart(2, '0')}
                </p>
                <p className="text-lg font-semibold text-gray-900 leading-snug whitespace-normal">{text}</p>
              </div>
            ))}
          </section>

          <div className="EvidenceLabelRow text-sm font-semibold text-gray-500 flex items-center gap-2" style={{ marginBottom: '8px' }}>
            <span className="text-lg">▼</span>
            このタイプの根拠
          </div>
          </div>
          <div className="BottomSection mt-6">
            <div
              className="EvidenceCard rounded-[28px] flex flex-col gap-3"
              style={{
                background: 'rgba(255,255,255,0.88)',
                padding: '24px 24px 28px',
                borderRadius: '28px',
                border: '1px solid rgba(255,255,255,0.8)',
                boxShadow: '0 12px 30px rgba(15,23,42,0.12)',
                backdropFilter: 'blur(16px)',
                width: '100%',
              }}
            >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex flex-col gap-1.5 min-w-0">
                <div className="EvidenceSectionTitle" style={{ marginBottom: '8px' }}>
                  <p className="text-sm font-semibold tracking-[0.05em] text-[#F082A9] uppercase">Tag Evidence</p>
                  <p className="text-lg font-bold text-gray-900 mt-1">タグの傾向</p>
                </div>
                <div className="flex flex-col gap-1.5">
                  {tagEvidence.categoryEvidence.map((category) => (
                    <div key={category.label} className="flex flex-col gap-1">
                      <div className="flex items-center justify-between text-sm text-gray-600">
                        <span>{category.label}</span>
                        <span className="font-semibold text-gray-900">{category.ratio}%</span>
                      </div>
                      <div className="h-1.5 bg-rose-50 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-rose-400 via-fuchsia-500 to-indigo-500"
                          style={{ width: `${category.ratio}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex flex-col gap-1.5">
                  {tagEvidence.tagTop.map((tag) => (
                    <div
                      key={tag.rank}
                      className="flex items-center justify-between px-3 py-2 rounded-xl bg-white/80 border border-white/70 shadow-sm"
                    >
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-gradient-to-br from-rose-400 to-fuchsia-500 text-white text-xs font-semibold flex items-center justify-center">
                          {tag.rank}
                        </span>
                        <div>
                          <p className="text-sm font-semibold text-gray-900">{tag.name}</p>
                          <p className="text-[11px] text-gray-400">{tag.sample}</p>
                        </div>
                      </div>
                      <p className="text-sm font-bold text-gray-900">{tag.ratio}</p>
                    </div>
                  ))}
                </div>
                <p className="EvidenceFooterText text-xs text-gray-500 truncate" style={{ marginTop: '4px', lineHeight: 1.2 }}>
                  {tagEvidence.comment}
                </p>
              </div>
              <div className="flex flex-col gap-1.5 min-w-0">
                <div className="EvidenceSectionTitle" style={{ marginBottom: '8px' }}>
                  <p className="text-sm font-semibold tracking-[0.05em] text-[#F082A9] uppercase">Actress Evidence</p>
                  <p className="text-lg font-bold text-gray-900 mt-1">よく見る出演者</p>
                </div>
                <div className="flex flex-wrap gap-3">
                  {featuredPerformers.length > 0 ? (
                    featuredPerformers.map((name) => (
                      <span
                        key={`pill-${name}`}
                        className="inline-flex items-center px-4 py-2 rounded-full bg-white shadow-[0_6px_16px_rgba(15,23,42,0.12)] border border-white/70 text-sm font-semibold text-gray-800"
                      >
                        {name}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-gray-400">まだ推し女優の傾向が出ていません。</span>
                  )}
                </div>
                <div className="flex flex-col gap-1.5">
                  {performerEvidence.performerTop.map((performer) => (
                    <div
                      key={`perf-${performer.rank}`}
                      className="flex items-center justify-between px-3 py-2 rounded-xl bg-white/80 border border-white/70 shadow-sm"
                    >
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-gradient-to-br from-rose-400 to-fuchsia-500 text-white text-xs font-semibold flex items-center justify-center">
                          {performer.rank}
                        </span>
                        <div>
                          <p className="text-sm font-semibold text-gray-900">{performer.name}</p>
                          <p className="text-[11px] text-gray-400">{performer.sample}</p>
                        </div>
                      </div>
                      <p className="text-sm font-bold text-gray-900">{performer.share}</p>
                    </div>
                  ))}
                </div>
                <p className="EvidenceFooterText text-xs text-gray-500 truncate" style={{ lineHeight: 1.2, marginTop: '4px' }}>
                  {performerEvidence.comment}
                </p>
              </div>
            </div>
            <div
              className="KeywordChips flex flex-wrap"
              style={{
                gap: '8px 12px',
                maxWidth: '640px',
              }}
            >
              {keywordPills.map((keyword) => (
                <span
                  key={`evidence-keyword-${keyword}`}
                  className="inline-flex items-center px-4 py-2 rounded-full bg-rose-50/80 text-rose-500 border border-rose-100 text-sm font-semibold shadow-inner"
                >
                  {keyword}
                </span>
              ))}
            </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default AnalysisShareCard;

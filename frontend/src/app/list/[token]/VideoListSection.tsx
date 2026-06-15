'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { GripVertical, Save, X, Plus, Loader2, Pencil, Trophy, List, Play } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { resolveThumbnail } from '@/utils/thumbnail';
import { resolveEmbedUrl } from '@/lib/videoMeta';

type Video = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  product_url: string | null;
  liked_at: string;
  source?: string | null;
  image_urls?: string[] | null;
  sort_order: number | null;
  section_id: string | null;
  sample_video_url?: string | null;
};

type Section = {
  id: string;
  title: string | null;
  display_mode: 'ranked' | 'plain';
  sort_order: number;
};

interface Props {
  ownerUserId: string;
  listId: string;
  videos: Video[];
  sections: Section[];
  affiliateUrls: Record<string, string>;
  listType: 'liked' | 'custom';
}

function toLgThumb(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

function getThumb(video: Video): string | null {
  const { primary } = resolveThumbnail({
    source: video.source,
    thumbnail_url: video.thumbnail_url,
    image_urls: video.image_urls,
  });
  return primary ?? toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
}

export default function VideoListSection({
  ownerUserId,
  listId,
  videos,
  sections: initialSections,
  affiliateUrls,
  listType,
}: Props) {
  const router = useRouter();
  const [isOwner, setIsOwner] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [addingSection, setAddingSection] = useState(false);
  const [playingVideo, setPlayingVideo] = useState<Video | null>(null);

  // セクション別に動画を管理（null = 未分類）
  const [videosBySection, setVideosBySection] = useState<Map<string | null, Video[]>>(new Map());
  const [sections, setSections] = useState<Section[]>([]);
  const [newSectionTitle, setNewSectionTitle] = useState('');

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setIsOwner(!!user && user.id === ownerUserId);
    });
  }, [ownerUserId]);

  const openEdit = useCallback(() => {
    const sorted = [...videos].sort((a, b) => {
      if (a.sort_order === null && b.sort_order === null) return 0;
      if (a.sort_order === null) return 1;
      if (b.sort_order === null) return -1;
      return a.sort_order - b.sort_order;
    });
    const secs = [...initialSections].sort((a, b) => a.sort_order - b.sort_order);

    const map = new Map<string | null, Video[]>();
    map.set(null, []);
    for (const s of secs) map.set(s.id, []);
    for (const v of sorted) {
      const key = v.section_id && map.has(v.section_id) ? v.section_id : null;
      map.get(key)!.push(v);
    }

    setVideosBySection(map);
    setSections(secs);
    setEditMode(true);
  }, [videos, initialSections]);

  // セクション並べ替え・動画D&Dを1つのhandlerで処理
  const onDragEnd = useCallback((result: DropResult) => {
    if (!result.destination) return;

    if (result.type === 'SECTION') {
      setSections(prev => {
        const next = [...prev];
        const [moved] = next.splice(result.source.index, 1);
        next.splice(result.destination!.index, 0, moved);
        return next;
      });
      return;
    }

    // type === 'VIDEO'
    const srcKey = result.source.droppableId === 'uncategorized' ? null : result.source.droppableId;
    const dstKey = result.destination.droppableId === 'uncategorized' ? null : result.destination.droppableId;

    setVideosBySection(prev => {
      const next = new Map(prev);
      if (srcKey === dstKey) {
        const list = [...(next.get(srcKey) ?? [])];
        const [moved] = list.splice(result.source.index, 1);
        list.splice(result.destination!.index, 0, moved);
        next.set(srcKey, list);
      } else {
        const srcList = [...(next.get(srcKey) ?? [])];
        const [moved] = srcList.splice(result.source.index, 1);
        next.set(srcKey, srcList);
        const dstList = [...(next.get(dstKey) ?? [])];
        dstList.splice(result.destination!.index, 0, { ...moved, section_id: dstKey });
        next.set(dstKey, dstList);
      }
      return next;
    });
  }, []);

  const toggleDisplayMode = useCallback((sectionId: string) => {
    setSections(prev =>
      prev.map(s =>
        s.id === sectionId
          ? { ...s, display_mode: s.display_mode === 'ranked' ? 'plain' : 'ranked' }
          : s
      )
    );
  }, []);

  const addSection = useCallback(async () => {
    if (!newSectionTitle.trim()) return;
    setAddingSection(true);
    const { data, error } = await supabase.rpc('upsert_list_section', {
      p_list_id: listId,
      p_section_id: null,
      p_title: newSectionTitle.trim(),
      p_sort_order: sections.length,
    });
    setAddingSection(false);
    if (error) { alert('セクション作成に失敗しました: ' + error.message); return; }
    const newId = data as string;
    setSections(prev => [...prev, { id: newId, title: newSectionTitle.trim(), display_mode: 'ranked', sort_order: prev.length }]);
    setVideosBySection(prev => { const next = new Map(prev); next.set(newId, []); return next; });
    setNewSectionTitle('');
  }, [newSectionTitle, listId, sections.length]);

  const deleteSection = useCallback(async (sectionId: string) => {
    const { error } = await supabase.rpc('delete_list_section', { p_section_id: sectionId });
    if (error) { alert('セクション削除に失敗しました: ' + error.message); return; }
    setSections(prev => prev.filter(s => s.id !== sectionId));
    setVideosBySection(prev => {
      const next = new Map(prev);
      const moved = (next.get(sectionId) ?? []).map(v => ({ ...v, section_id: null }));
      next.delete(sectionId);
      next.set(null, [...(next.get(null) ?? []), ...moved]);
      return next;
    });
  }, []);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      // フラットな順序リストを構築（セクション順 → 未分類）
      const flatVideos: Video[] = [];
      for (const section of sections) {
        for (const v of videosBySection.get(section.id) ?? []) flatVideos.push(v);
      }
      for (const v of videosBySection.get(null) ?? []) flatVideos.push(v);

      const { error: reorderErr } = await supabase.rpc('reorder_list_videos', {
        p_list_id: listId,
        p_video_ids: flatVideos.map(v => v.id),
      });
      if (reorderErr) throw reorderErr;

      // セクション割り当て変更分を保存
      for (const [sectionId, vids] of videosBySection) {
        for (const video of vids) {
          const original = videos.find(v => v.id === video.id);
          if (original?.section_id !== sectionId) {
            const { error } = await supabase.rpc('assign_video_section', {
              p_list_id: listId,
              p_video_id: video.id,
              p_section_id: sectionId,
            });
            if (error) throw error;
          }
        }
      }

      // セクション並べ替え
      if (sections.length > 0) {
        const { error } = await supabase.rpc('reorder_list_sections', {
          p_list_id: listId,
          p_section_ids: sections.map(s => s.id),
        });
        if (error) throw error;
      }

      // display_mode 変更分を保存
      for (const section of sections) {
        const original = initialSections.find(s => s.id === section.id);
        if (original && original.display_mode !== section.display_mode) {
          const { error } = await supabase.rpc('upsert_list_section', {
            p_list_id: listId,
            p_section_id: section.id,
            p_title: section.title ?? '',
            p_sort_order: section.sort_order,
            p_display_mode: section.display_mode,
          });
          if (error) throw error;
        }
      }

      setEditMode(false);
      router.refresh();
    } catch (err: unknown) {
      alert('保存に失敗しました: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setSaving(false);
    }
  }, [videosBySection, sections, initialSections, listId, videos, router]);

  // ---- 通常表示（閲覧モード） ----
  if (!editMode) {
    const isFlat = initialSections.length === 0;
    const flatIsRanked = listType === 'custom' && videos.some(v => v.sort_order !== null);

    const sectionMap = new Map<string | null, Video[]>();
    sectionMap.set(null, []);
    for (const s of initialSections) sectionMap.set(s.id, []);
    for (const v of videos) {
      const key = v.section_id ?? null;
      if (!sectionMap.has(key)) sectionMap.set(key, []);
      sectionMap.get(key)!.push(v);
    }

    return (
      <div>
        {isOwner && listType === 'custom' && (
          <div className="mb-4 flex justify-end">
            <button
              onClick={openEdit}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#30363d] hover:border-[#8b949e] text-[#8b949e] hover:text-[#e6edf3] text-xs font-semibold transition-colors"
            >
              <Pencil size={12} />
              並べ替え・セクション
            </button>
          </div>
        )}

        {videos.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだ作品がありません。</p>
        ) : isFlat ? (
          <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
            {videos.map((video, i) => (
              <NormalCard
                key={video.id}
                video={video}
                affiliateUrl={affiliateUrls[video.id] ?? ''}
                rank={flatIsRanked ? i + 1 : null}
                onPlay={setPlayingVideo}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-10">
            {initialSections.map(section => {
              const sv = sectionMap.get(section.id) ?? [];
              if (sv.length === 0) return null;
              return (
                <div key={section.id}>
                  {section.title && (
                    <h2 className="text-sm font-bold text-[#8b949e] uppercase tracking-wider mb-4 pb-2 border-b border-[#21262d]">
                      {section.title}
                    </h2>
                  )}
                  <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
                    {sv.map((video, i) => (
                      <NormalCard
                        key={video.id}
                        video={video}
                        affiliateUrl={affiliateUrls[video.id] ?? ''}
                        rank={section.display_mode === 'ranked' ? i + 1 : null}
                        onPlay={setPlayingVideo}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
            {(sectionMap.get(null) ?? []).length > 0 && (
              <div>
                <h2 className="text-sm font-bold text-[#484f58] uppercase tracking-wider mb-4 pb-2 border-b border-[#21262d]">
                  未分類
                </h2>
                <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
                  {(sectionMap.get(null) ?? []).map(video => (
                    <NormalCard
                      key={video.id}
                      video={video}
                      affiliateUrl={affiliateUrls[video.id] ?? ''}
                      rank={null}
                      onPlay={setPlayingVideo}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* サンプル再生モーダル */}
        {playingVideo && (
          <SampleModal
            video={playingVideo}
            affiliateUrl={affiliateUrls[playingVideo.id] ?? ''}
            onClose={() => setPlayingVideo(null)}
          />
        )}
      </div>
    );
  }

  // ---- 編集モード ----
  const uncategorized = videosBySection.get(null) ?? [];

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      {/* ヘッダー */}
      <div className="sticky top-0 z-10 mb-4 flex items-center justify-between gap-3 px-4 py-3 rounded-xl bg-violet-500/10 border border-violet-500/40 backdrop-blur">
        <span className="text-sm font-bold text-violet-300">並べ替え・セクション編集</span>
        <div className="flex items-center gap-2">
          <button
            onClick={save}
            disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-60 text-white text-xs font-bold transition-colors"
          >
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
            保存
          </button>
          <button
            onClick={() => setEditMode(false)}
            className="p-1.5 rounded-lg hover:bg-[#21262d] text-[#656d76] hover:text-[#e6edf3] transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* セクション（D&D対応） */}
      <Droppable droppableId="sections" type="SECTION">
        {provided => (
          <div ref={provided.innerRef} {...provided.droppableProps} className="space-y-4 mb-4">
            {sections.map((section, sIdx) => {
              const sectionVideos = videosBySection.get(section.id) ?? [];
              return (
                <Draggable key={section.id} draggableId={`sec-${section.id}`} index={sIdx}>
                  {(provided, snapshot) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      className={`rounded-xl border bg-[#0d1117] transition-shadow ${
                        snapshot.isDragging ? 'border-violet-500/50 shadow-lg' : 'border-[#21262d]'
                      }`}
                    >
                      {/* セクションヘッダー */}
                      <div className="flex items-center gap-2 px-3 py-2 border-b border-[#21262d]">
                        <div {...provided.dragHandleProps} className="text-[#484f58] hover:text-[#8b949e] cursor-grab shrink-0">
                          <GripVertical size={14} />
                        </div>
                        <span className="text-xs font-bold text-[#e6edf3] flex-1">{section.title ?? '無題'}</span>
                        <button
                          onClick={() => toggleDisplayMode(section.id)}
                          title={section.display_mode === 'ranked' ? 'ランキング表示' : 'プレーン表示'}
                          className={`flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-bold transition-colors ${
                            section.display_mode === 'ranked'
                              ? 'border-amber-500/40 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20'
                              : 'border-[#30363d] bg-[#0d1117] text-[#656d76] hover:border-[#8b949e] hover:text-[#8b949e]'
                          }`}
                        >
                          {section.display_mode === 'ranked' ? <><Trophy size={9} />ランク</> : <><List size={9} />プレーン</>}
                        </button>
                        <button onClick={() => deleteSection(section.id)} className="text-[#484f58] hover:text-red-400 transition-colors shrink-0">
                          <X size={12} />
                        </button>
                      </div>

                      {/* セクション内動画 */}
                      <Droppable droppableId={section.id} type="VIDEO" direction="horizontal">
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.droppableProps}
                            className={`p-2 min-h-[80px] grid grid-cols-4 sm:grid-cols-6 gap-1.5 content-start transition-colors rounded-b-xl ${
                              snapshot.isDraggingOver ? 'bg-violet-500/5' : ''
                            }`}
                          >
                            {sectionVideos.map((video, vIdx) => (
                              <CompactCard
                                key={video.id}
                                video={video}
                                index={vIdx}
                                isRanked={section.display_mode === 'ranked'}
                              />
                            ))}
                            {provided.placeholder}
                            {sectionVideos.length === 0 && !snapshot.isDraggingOver && (
                              <p className="col-span-4 sm:col-span-6 text-[10px] text-[#484f58] py-2 text-center">ここにドラッグして追加</p>
                            )}
                          </div>
                        )}
                      </Droppable>
                    </div>
                  )}
                </Draggable>
              );
            })}
            {provided.placeholder}
          </div>
        )}
      </Droppable>

      {/* セクション追加 */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={newSectionTitle}
          onChange={e => setNewSectionTitle(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addSection()}
          placeholder="新しいセクション名を追加..."
          className="flex-1 px-3 py-1.5 rounded-lg bg-[#161b22] border border-[#30363d] focus:border-violet-500/60 outline-none text-xs text-[#e6edf3] placeholder-[#484f58]"
        />
        <button
          onClick={addSection}
          disabled={addingSection || !newSectionTitle.trim()}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-violet-300 disabled:opacity-40 text-xs font-semibold transition-colors"
        >
          {addingSection ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
          追加
        </button>
      </div>

      {/* 未分類 */}
      <div className="rounded-xl border border-[#21262d] bg-[#0d1117]">
        <div className="px-3 py-2 border-b border-[#21262d]">
          <span className="text-xs font-bold text-[#484f58]">未分類</span>
        </div>
        <Droppable droppableId="uncategorized" type="VIDEO" direction="horizontal">
          {(provided, snapshot) => (
            <div
              ref={provided.innerRef}
              {...provided.droppableProps}
              className={`p-2 min-h-[80px] grid grid-cols-4 sm:grid-cols-6 gap-1.5 content-start transition-colors rounded-b-xl ${
                snapshot.isDraggingOver ? 'bg-violet-500/5' : ''
              }`}
            >
              {uncategorized.map((video, vIdx) => (
                <CompactCard key={video.id} video={video} index={vIdx} isRanked={false} />
              ))}
              {provided.placeholder}
              {uncategorized.length === 0 && !snapshot.isDraggingOver && (
                <p className="col-span-4 sm:col-span-6 text-[10px] text-[#484f58] py-2 text-center">未分類の作品はありません</p>
              )}
            </div>
          )}
        </Droppable>
      </div>
    </DragDropContext>
  );
}

function CompactCard({ video, index, isRanked }: { video: Video; index: number; isRanked: boolean }) {
  const thumb = getThumb(video);
  return (
    <Draggable draggableId={video.id} index={index}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          title={video.title ?? ''}
          className={`relative rounded overflow-hidden border bg-[#161b22] cursor-grab active:cursor-grabbing transition-all ${
            snapshot.isDragging
              ? 'border-violet-500/60 shadow-lg shadow-violet-500/20 rotate-1 scale-105'
              : 'border-[#30363d] hover:border-[#8b949e]'
          }`}
        >
          {thumb ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={thumb} alt="" className="w-full aspect-video object-cover" loading="lazy" />
          ) : (
            <div className="w-full aspect-video bg-[#21262d] flex items-center justify-center">
              <span className="text-[#484f58] text-[9px]">No Image</span>
            </div>
          )}
          {isRanked && (
            <div className="absolute top-0.5 left-0.5 min-w-[16px] h-[16px] px-0.5 rounded bg-black/70 flex items-center justify-center">
              <span className="text-[9px] font-black text-violet-300">#{index + 1}</span>
            </div>
          )}
          <div className="absolute top-0.5 right-0.5 text-[#8b949e]/60">
            <GripVertical size={10} />
          </div>
        </div>
      )}
    </Draggable>
  );
}

function NormalCard({ video, affiliateUrl, rank, onPlay }: {
  video: Video;
  affiliateUrl: string;
  rank: number | null;
  onPlay: (v: Video) => void;
}) {
  const thumb = getThumb(video);
  const hasSample = !!video.sample_video_url && video.source !== 'vr';

  const cardContent = (
    <>
      <div className="relative">
        {thumb ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumb} alt={video.title ?? ''} className="w-full h-auto group-hover:opacity-90 transition-opacity" loading="lazy" />
        ) : (
          <div className="w-full aspect-video bg-[#21262d] flex items-center justify-center">
            <span className="text-[#484f58] text-xs">No Image</span>
          </div>
        )}
        {rank !== null && (
          <div className="absolute top-1.5 left-1.5 min-w-[22px] h-[22px] px-1.5 rounded-md bg-black/70 backdrop-blur flex items-center justify-center">
            <span className="text-[11px] font-black text-violet-300">#{rank}</span>
          </div>
        )}
        {hasSample && (
          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30">
            <div className="w-10 h-10 flex items-center justify-center rounded-full bg-black/70">
              <Play size={18} className="text-white ml-0.5" fill="white" />
            </div>
          </div>
        )}
      </div>
      <div className="p-2">
        <p className="text-xs text-[#8b949e] leading-tight line-clamp-2">{video.title ?? ''}</p>
      </div>
    </>
  );

  return (
    <div className="group relative mb-3 break-inside-avoid">
      {hasSample ? (
        <button
          onClick={() => onPlay(video)}
          className="w-full text-left rounded-lg overflow-hidden border border-[#21262d] hover:border-violet-500/50 transition-colors bg-[#161b22]"
        >
          {cardContent}
        </button>
      ) : (
        <a
          href={affiliateUrl || '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="block rounded-lg overflow-hidden border border-[#21262d] hover:border-violet-500/50 transition-colors bg-[#161b22]"
        >
          {cardContent}
        </a>
      )}
    </div>
  );
}

function SampleModal({ video, affiliateUrl, onClose }: { video: Video; affiliateUrl: string; onClose: () => void }) {
  // ユーザーがPlay▶をクリックして初めて再生開始
  const [started, setStarted] = useState(false);
  const overlayTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thumb = getThumb(video);
  const embed = resolveEmbedUrl({
    source: video.source,
    externalId: video.external_id,
    sampleVideoUrl: video.sample_video_url,
  });

  useEffect(() => {
    return () => { if (overlayTimer.current) clearTimeout(overlayTimer.current); };
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-lg rounded-2xl overflow-hidden bg-[#0d1117] shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-2 right-2 z-20 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80"
        >
          <X size={16} />
        </button>

        <div className="relative w-full aspect-video bg-black overflow-hidden">
          {/* サムネイル＋Playボタン（未再生時） */}
          {!started && (
            <div
              className="absolute inset-0 z-10 cursor-pointer flex items-center justify-center"
              style={{
                backgroundImage: thumb ? `url(${thumb})` : undefined,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundColor: '#111',
              }}
              onClick={() => setStarted(true)}
            >
              <div className="absolute inset-0 bg-black/40" />
              <div className="relative z-10 w-16 h-16 flex items-center justify-center rounded-full bg-black/70 hover:bg-black/90 transition-colors">
                <Play className="text-white w-8 h-8 ml-1" fill="white" />
              </div>
            </div>
          )}

          {/* 再生コンテンツ（startedになったら描画） */}
          {started && embed?.type === 'mp4' && (
            <video
              src={embed.url}
              controls
              autoPlay
              playsInline
              className="absolute inset-0 w-full h-full bg-black"
            />
          )}
          {started && embed?.type === 'iframe' && (
            <iframe
              src={embed.url}
              frameBorder="0"
              scrolling="no"
              referrerPolicy="no-referrer"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
              loading="eager"
              className="absolute inset-0 w-full h-full"
            />
          )}
        </div>

        <div className="p-3 flex items-center justify-between gap-3">
          <p className="text-xs text-[#8b949e] line-clamp-2 flex-1">{video.title ?? ''}</p>
          {affiliateUrl && (
            <a
              href={affiliateUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors"
            >
              本編を見る →
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

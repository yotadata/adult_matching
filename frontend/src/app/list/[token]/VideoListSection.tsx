'use client';

import { useState, useCallback, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { GripVertical, Save, X, Plus, Loader2, Pencil, Trophy, List } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { resolveThumbnail } from '@/utils/thumbnail';

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

  const [orderedVideos, setOrderedVideos] = useState<Video[]>([]);
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
    setOrderedVideos(sorted);
    setSections([...initialSections].sort((a, b) => a.sort_order - b.sort_order));
    setEditMode(true);
  }, [videos, initialSections]);

  const onVideoDragEnd = useCallback((result: DropResult) => {
    if (!result.destination) return;
    setOrderedVideos(prev => {
      const next = [...prev];
      const [moved] = next.splice(result.source.index, 1);
      next.splice(result.destination!.index, 0, moved);
      return next;
    });
  }, []);

  const onSectionDragEnd = useCallback((result: DropResult) => {
    if (!result.destination) return;
    setSections(prev => {
      const next = [...prev];
      const [moved] = next.splice(result.source.index, 1);
      next.splice(result.destination!.index, 0, moved);
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

  const assignSection = useCallback((videoId: string, sectionId: string | null) => {
    setOrderedVideos(prev =>
      prev.map(v => v.id === videoId ? { ...v, section_id: sectionId } : v)
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
    setSections(prev => [...prev, {
      id: data as string,
      title: newSectionTitle.trim(),
      display_mode: 'ranked',
      sort_order: prev.length,
    }]);
    setNewSectionTitle('');
  }, [newSectionTitle, listId, sections.length]);

  const deleteSection = useCallback(async (sectionId: string) => {
    const { error } = await supabase.rpc('delete_list_section', { p_section_id: sectionId });
    if (error) { alert('セクション削除に失敗しました: ' + error.message); return; }
    setSections(prev => prev.filter(s => s.id !== sectionId));
    setOrderedVideos(prev => prev.map(v => v.section_id === sectionId ? { ...v, section_id: null } : v));
  }, []);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      // 動画並べ替え
      const { error: reorderErr } = await supabase.rpc('reorder_list_videos', {
        p_list_id: listId,
        p_video_ids: orderedVideos.map(v => v.id),
      });
      if (reorderErr) throw reorderErr;

      // 動画セクション割り当て
      for (const video of orderedVideos) {
        const original = videos.find(v => v.id === video.id);
        if (original?.section_id !== video.section_id) {
          const { error } = await supabase.rpc('assign_video_section', {
            p_list_id: listId,
            p_video_id: video.id,
            p_section_id: video.section_id,
          });
          if (error) throw error;
        }
      }

      // セクション並べ替え
      if (sections.length > 0) {
        const { error: secReorderErr } = await supabase.rpc('reorder_list_sections', {
          p_list_id: listId,
          p_section_ids: sections.map(s => s.id),
        });
        if (secReorderErr) throw secReorderErr;
      }

      // display_mode 更新（変更があったセクションのみ）
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
  }, [orderedVideos, sections, initialSections, listId, videos, router]);

  // ---- 通常表示（閲覧モード） ----
  if (!editMode) {
    const flatVideos = videos;
    const sectionsArr = initialSections;
    const isFlat = sectionsArr.length === 0;
    const flatIsRanked = listType === 'custom' && flatVideos.some(v => v.sort_order !== null);

    const sectionMap = new Map<string | null, Video[]>();
    sectionMap.set(null, []);
    for (const s of sectionsArr) sectionMap.set(s.id, []);
    for (const v of flatVideos) {
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

        {flatVideos.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだ作品がありません。</p>
        ) : isFlat ? (
          <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
            {flatVideos.map((video, i) => (
              <NormalCard
                key={video.id}
                video={video}
                affiliateUrl={affiliateUrls[video.id] ?? ''}
                rank={flatIsRanked ? i + 1 : null}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-10">
            {sectionsArr.map(section => {
              const sv = sectionMap.get(section.id) ?? [];
              const isRanked = section.display_mode === 'ranked';
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
                        rank={isRanked ? i + 1 : null}
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
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // ---- 編集モード（D&D） ----
  return (
    <div>
      {/* 編集モードヘッダー */}
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

      {/* セクション管理 */}
      <div className="mb-5 space-y-2">
        <DragDropContext onDragEnd={onSectionDragEnd}>
          <Droppable droppableId="section-list" direction="vertical">
            {provided => (
              <div ref={provided.innerRef} {...provided.droppableProps} className="space-y-1.5">
                {sections.map((section, i) => (
                  <Draggable key={section.id} draggableId={`sec-${section.id}`} index={i}>
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-[#161b22] border text-xs transition-shadow ${
                          snapshot.isDragging ? 'border-violet-500/50 shadow-md' : 'border-[#30363d]'
                        }`}
                      >
                        <div {...provided.dragHandleProps} className="text-[#484f58] hover:text-[#8b949e] cursor-grab">
                          <GripVertical size={13} />
                        </div>
                        <span className="text-[#e6edf3] flex-1">{section.title ?? '無題'}</span>
                        {/* display_mode トグル */}
                        <button
                          onClick={() => toggleDisplayMode(section.id)}
                          title={section.display_mode === 'ranked' ? 'ランキング表示（クリックでプレーンに変更）' : 'プレーン表示（クリックでランキングに変更）'}
                          className={`flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-bold transition-colors ${
                            section.display_mode === 'ranked'
                              ? 'border-amber-500/40 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20'
                              : 'border-[#30363d] bg-[#0d1117] text-[#656d76] hover:border-[#8b949e] hover:text-[#8b949e]'
                          }`}
                        >
                          {section.display_mode === 'ranked'
                            ? <><Trophy size={9} />ランク</>
                            : <><List size={9} />プレーン</>
                          }
                        </button>
                        <button
                          onClick={() => deleteSection(section.id)}
                          className="text-[#484f58] hover:text-red-400 transition-colors"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
        {/* 新規セクション追加 */}
        <div className="flex gap-2">
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
      </div>

      {/* 動画 D&D リスト */}
      <DragDropContext onDragEnd={onVideoDragEnd}>
        <Droppable droppableId="video-list">
          {provided => (
            <div
              ref={provided.innerRef}
              {...provided.droppableProps}
              className="grid grid-cols-2 sm:grid-cols-3 gap-3"
            >
              {orderedVideos.map((video, i) => {
                const { primary: resolvedThumb } = resolveThumbnail({
                  source: video.source,
                  thumbnail_url: video.thumbnail_url,
                  image_urls: video.image_urls,
                });
                const thumb = resolvedThumb ?? toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);

                return (
                  <Draggable key={video.id} draggableId={video.id} index={i}>
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        className={`relative rounded-lg overflow-hidden border bg-[#161b22] transition-shadow ${
                          snapshot.isDragging
                            ? 'border-violet-500/60 shadow-lg shadow-violet-500/20 rotate-1'
                            : 'border-[#30363d]'
                        }`}
                      >
                        {/* ドラッグハンドル */}
                        <div
                          {...provided.dragHandleProps}
                          className="absolute top-1.5 right-1.5 z-10 p-1 rounded bg-black/60 backdrop-blur cursor-grab active:cursor-grabbing text-[#8b949e] hover:text-white transition-colors"
                        >
                          <GripVertical size={14} />
                        </div>

                        {/* ランク番号 */}
                        <div className="absolute top-1.5 left-1.5 z-10 min-w-[20px] h-[20px] px-1 rounded bg-black/70 backdrop-blur flex items-center justify-center">
                          <span className="text-[10px] font-black text-violet-300">#{i + 1}</span>
                        </div>

                        {/* サムネイル */}
                        {thumb ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={thumb}
                            alt={video.title ?? ''}
                            className="w-full h-auto"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full aspect-video bg-[#21262d] flex items-center justify-center">
                            <span className="text-[#484f58] text-xs">No Image</span>
                          </div>
                        )}

                        {/* タイトル + セクション選択 */}
                        <div className="p-2 space-y-1.5">
                          <p className="text-[11px] text-[#8b949e] leading-tight line-clamp-2">
                            {video.title ?? ''}
                          </p>
                          {sections.length > 0 && (
                            <select
                              value={video.section_id ?? ''}
                              onChange={e => assignSection(video.id, e.target.value || null)}
                              className="w-full text-[11px] bg-[#0d1117] border border-[#30363d] rounded px-1.5 py-0.5 text-[#8b949e]"
                            >
                              <option value="">未分類</option>
                              {sections.map(s => (
                                <option key={s.id} value={s.id}>{s.title ?? '無題'}</option>
                              ))}
                            </select>
                          )}
                        </div>
                      </div>
                    )}
                  </Draggable>
                );
              })}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );
}

function NormalCard({ video, affiliateUrl, rank }: {
  video: Video;
  affiliateUrl: string;
  rank: number | null;
}) {
  const { primary: resolvedThumb } = resolveThumbnail({
    source: video.source,
    thumbnail_url: video.thumbnail_url,
    image_urls: video.image_urls,
  });
  const thumb = resolvedThumb ?? toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);

  return (
    <div className="group relative mb-3 break-inside-avoid">
      <a
        href={affiliateUrl || '#'}
        target="_blank"
        rel="noopener noreferrer"
        className="block rounded-lg overflow-hidden border border-[#21262d] hover:border-violet-500/50 transition-colors bg-[#161b22]"
      >
        <div className="relative">
          {thumb ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={thumb}
              alt={video.title ?? ''}
              className="w-full h-auto group-hover:opacity-90 transition-opacity"
              loading="lazy"
            />
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
        </div>
        <div className="p-2">
          <p className="text-xs text-[#8b949e] leading-tight line-clamp-2">
            {video.title ?? ''}
          </p>
        </div>
      </a>
    </div>
  );
}

'use client';

import { useEffect, useMemo, useState } from 'react';
import { supabase } from '@/lib/supabase';
import Image from 'next/image';
import Link from 'next/link';

type Performer = { id: string; name: string };
type Tag = { id: string; name: string };

type LikedVideo = {
  id: string;
  external_id: string;
  title: string;
  description?: string | null;
  thumbnail_url?: string | null;
  product_url?: string | null;
  price?: number | null;
  product_released_at?: string | null;
  liked_at: string;
  performers: Performer[];
  tags: Tag[];
};

const pageSize = 24;

export default function LikesPage() {
  const [items, setItems] = useState<LikedVideo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);

  const [q, setQ] = useState('');
  const [sort, setSort] = useState<'liked_at' | 'released' | 'price' | 'title'>('liked_at');
  const [order, setOrder] = useState<'desc' | 'asc'>('desc');
  const [priceMin, setPriceMin] = useState<string>('');
  const [priceMax, setPriceMax] = useState<string>('');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [page, setPage] = useState(0);

  const params = useMemo(() => ({ q, sort, order, priceMin, priceMax, dateFrom, dateTo, page }), [q, sort, order, priceMin, priceMax, dateFrom, dateTo, page]);

  useEffect(() => {
    const check = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(!!session?.user);
      if (!session?.user) return;
    };
    check();
  }, []);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session?.user) {
          setItems([]);
          setLoading(false);
          return;
        }
        const { data, error } = await supabase.rpc('get_user_likes', {
          p_search: q || null,
          p_sort: sort,
          p_order: order,
          p_limit: pageSize,
          p_offset: page * pageSize,
          p_price_min: priceMin ? Number(priceMin) : null,
          p_price_max: priceMax ? Number(priceMax) : null,
          p_release_gte: dateFrom ? new Date(dateFrom).toISOString() : null,
          p_release_lte: dateTo ? new Date(dateTo).toISOString() : null,
          p_tag_ids: null,
          p_performer_ids: null,
        });
        if (error) throw error;
        setItems((data as LikedVideo[]) || []);
      } catch (e: unknown) {
        let message = '読み込みに失敗しました';
        if (typeof e === 'string') message = e;
        else if (e && typeof e === 'object' && 'message' in e && typeof (e as { message?: unknown }).message === 'string') {
          message = (e as { message: string }).message;
        }
        setError(message);
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [q, sort, order, priceMin, priceMax, dateFrom, dateTo, page]);

  if (isLoggedIn === false) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-10 text-center">
        <p className="text-gray-700 mb-4">いいね一覧の表示にはログインが必要です。</p>
        <button
          className="inline-block bg-violet-500 text-white font-bold rounded-lg px-4 py-2"
          onClick={() => { try { window.dispatchEvent(new Event('open-register-modal')); } catch {} }}
        >
          新規登録 / ログイン
        </button>
      </div>
    );
  }

  return (
    <main className="max-w-6xl mx-auto px-4 py-6">
      <h1 className="text-xl font-bold mb-4">いいね一覧</h1>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 mb-4">
        <input
          className="md:col-span-4 border rounded-lg px-3 py-2"
          placeholder="タイトルで検索"
          value={q}
          onChange={(e) => { setPage(0); setQ(e.target.value); }}
        />
        <div className="md:col-span-3 flex items-center gap-2">
          <label className="text-sm text-gray-600">並び替え</label>
          <select className="border rounded-lg px-2 py-2" value={sort} onChange={(e) => { setPage(0); setSort(e.target.value as 'liked_at' | 'released' | 'price' | 'title'); }}>
            <option value="liked_at">いいね日時</option>
            <option value="released">発売日</option>
            <option value="price">価格</option>
            <option value="title">タイトル</option>
          </select>
          <select className="border rounded-lg px-2 py-2" value={order} onChange={(e) => { setPage(0); setOrder(e.target.value as 'desc' | 'asc'); }}>
            <option value="desc">降順</option>
            <option value="asc">昇順</option>
          </select>
        </div>
        <div className="md:col-span-5 grid grid-cols-2 gap-2">
          <input
            className="border rounded-lg px-3 py-2"
            type="number"
            placeholder="最低価格"
            value={priceMin}
            onChange={(e) => { setPage(0); setPriceMin(e.target.value); }}
          />
          <input
            className="border rounded-lg px-3 py-2"
            type="number"
            placeholder="最高価格"
            value={priceMax}
            onChange={(e) => { setPage(0); setPriceMax(e.target.value); }}
          />
          <input
            className="border rounded-lg px-3 py-2"
            type="date"
            placeholder="発売日(自)"
            value={dateFrom}
            onChange={(e) => { setPage(0); setDateFrom(e.target.value); }}
          />
          <input
            className="border rounded-lg px-3 py-2"
            type="date"
            placeholder="発売日(至)"
            value={dateTo}
            onChange={(e) => { setPage(0); setDateTo(e.target.value); }}
          />
        </div>
      </div>

      {error && <div className="mb-3 text-sm text-red-600">{error}</div>}

      {/* Grid */}
      {loading ? (
        <div className="py-10 text-center text-gray-500">読み込み中...</div>
      ) : items.length === 0 ? (
        <div className="py-10 text-center text-gray-500">該当する動画がありません</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {items.map((v) => (
            <div key={v.id} className="bg-white rounded-lg shadow-sm overflow-hidden border">
              <Link href={`/videos/${v.id}`} className="block relative w-full" style={{ paddingBottom: '60%' }}>
                {v.thumbnail_url ? (
                  <Image src={v.thumbnail_url} alt={v.title} fill className="object-cover" />
                ) : (
                  <div className="absolute inset-0 bg-gray-200" />
                )}
              </Link>
              <div className="p-3">
                <Link href={`/videos/${v.id}`} className="block font-semibold text-sm line-clamp-2 mb-1">
                  {v.title}
                </Link>
                <div className="text-xs text-gray-500 flex justify-between items-center">
                  <span>{v.product_released_at ? new Date(v.product_released_at).toLocaleDateString('ja-JP') : '-'}</span>
                  <span>{v.price != null ? `￥${Number(v.price).toLocaleString()}` : '-'}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      <div className="flex justify-center gap-3 mt-6">
        <button
          className="px-3 py-2 rounded-lg border disabled:opacity-50"
          disabled={page === 0}
          onClick={() => setPage((p) => Math.max(0, p - 1))}
        >
          前へ
        </button>
        <button
          className="px-3 py-2 rounded-lg border"
          onClick={() => setPage((p) => p + 1)}
        >
          次へ
        </button>
      </div>
    </main>
  );
}

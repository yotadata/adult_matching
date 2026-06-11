# キュレーター機能 MVP 仕様書

## 目的

「人を軸にしたリスト消費」と「将来的なキュレーター制度の検証」を行う。
楽天ROOMやSpotifyプレイリストのように、「この人のおすすめだから見たい」という体験を作る。

**MVPで検証する仮説**
- ユーザーはリスト作成者に興味を持つか
- 特定ユーザーのリストを複数閲覧するか
- 人気リストを判別できるか（閲覧数・いいね）
- 将来的なキュレーター制度の土台として利用できるか

## 定義

**キュレーター = リストを公開しているユーザー**

新エンティティは作らない。既存の `auth.users` をそのまま利用する。

---

## 現状のデータ構造（実装済み）

### profiles ビュー（`supabase/migrations/20260609063622_update_profiles_view_add_username.sql`）

```sql
CREATE VIEW public.profiles AS
SELECT
  id AS user_id,
  COALESCE(raw_user_meta_data->>'display_name', '') AS display_name,
  raw_user_meta_data->>'username' AS username,
  created_at
FROM auth.users;
```

→ `avatar_url`, `bio`, `x_url`, `affiliate_*` は未実装。今回追加する。

### lists テーブル

`token`（公開URL用）, `user_id`, `title`, `list_type`, `is_public` などを持つ。
→ `view_count` は未実装。今回追加する。

### list_likes テーブル

未実装。今回追加する。

---

## 実装タスク一覧

### 1. DB: ユーザープロフィール拡張

**マイグレーション**: `supabase/migrations/<timestamp>_add_curator_profile_fields.sql`

`auth.users.raw_user_meta_data` にフィールドを追加するのではなく、
専用の `public.user_profiles` テーブルを作成して拡張情報を管理する。

```sql
CREATE TABLE public.user_profiles (
  user_id       uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  avatar_url    text,
  bio           text,
  x_url         text,
  affiliate_fanza_id  text,
  affiliate_fc2_id    text,
  affiliate_mgs_id    text,
  updated_at    timestamptz DEFAULT now()
);
-- RLS: 本人のみ update、全員 select 可
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "public read" ON public.user_profiles FOR SELECT USING (true);
CREATE POLICY "owner update" ON public.user_profiles FOR ALL USING (auth.uid() = user_id);
```

`public.profiles` ビューを更新して `user_profiles` を LEFT JOIN する。

---

### 2. DB: リスト閲覧数

**マイグレーション**: `supabase/migrations/<timestamp>_add_list_view_count.sql`

```sql
ALTER TABLE public.lists ADD COLUMN view_count bigint NOT NULL DEFAULT 0;
```

インクリメント用 RPC:

```sql
CREATE OR REPLACE FUNCTION public.increment_list_view(p_list_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  UPDATE public.lists SET view_count = view_count + 1 WHERE list_id = p_list_id;
END;
$$;
```

- 初期はPVベース（ページ表示のたびにインクリメント）
- ユニークユーザー集計は不要

---

### 3. DB: リストいいね

**マイグレーション**: `supabase/migrations/<timestamp>_add_list_likes.sql`

```sql
CREATE TABLE public.list_likes (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  list_id     uuid NOT NULL REFERENCES public.lists(list_id) ON DELETE CASCADE,
  fingerprint text NOT NULL,  -- ブラウザフィンガープリントまたはCookie ID
  created_at  timestamptz DEFAULT now(),
  UNIQUE(list_id, fingerprint)
);
ALTER TABLE public.list_likes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "public insert" ON public.list_likes FOR INSERT WITH CHECK (true);
CREATE POLICY "public read" ON public.list_likes FOR SELECT USING (true);
```

いいね数取得・トグル用 RPC:

```sql
CREATE OR REPLACE FUNCTION public.toggle_list_like(p_list_id uuid, p_fingerprint text)
RETURNS json LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_liked boolean;
  v_count bigint;
BEGIN
  IF EXISTS (SELECT 1 FROM public.list_likes WHERE list_id = p_list_id AND fingerprint = p_fingerprint) THEN
    DELETE FROM public.list_likes WHERE list_id = p_list_id AND fingerprint = p_fingerprint;
    v_liked := false;
  ELSE
    INSERT INTO public.list_likes(list_id, fingerprint) VALUES (p_list_id, p_fingerprint);
    v_liked := true;
  END IF;
  SELECT COUNT(*) INTO v_count FROM public.list_likes WHERE list_id = p_list_id;
  RETURN json_build_object('liked', v_liked, 'count', v_count);
END;
$$;
```

- 認証必須にしない（Cookie/fingerprint で簡易重複防止）

---

### 4. フロントエンド: 公開プロフィールページ

**ファイル**: `frontend/src/app/u/[username]/page.tsx`（既存ディレクトリを利用）

表示内容:
- アイコン（`avatar_url` → なければデフォルトアバター）
- ユーザー名（`username`）
- 自己紹介（`bio`）
- Xリンク（`x_url`）
- 公開リスト一覧（`is_public = true`）
- 公開リスト数

URL例: `/u/mashironcake`

---

### 5. フロントエンド: リスト詳細ページに作成者情報を表示

**ファイル**: `frontend/src/app/list/[token]/page.tsx`（既存）

追加する表示:
- 作成者アイコン + ユーザー名（`/u/<username>` へのリンク）
- 閲覧数 `👁 3,251`
- いいね `❤️ 84`（トグルボタン）

ページ表示時に `increment_list_view` RPC を呼ぶ。
fingerprint は `localStorage` に UUID を保存して代替する。

---

### 6. フロントエンド: プロフィール編集UI

**ファイル**: `frontend/src/app/account-management/` 配下に追加

編集可能フィールド:
- アイコン画像（Supabase Storage へアップロード）
- 自己紹介
- XのURL
- FANZAアフィリエイトID
- FC2アフィリエイトID
- MGSアフィリエイトID

---

### 7. アフィリエイトリンク生成ロジック変更

**ファイル**: `frontend/src/app/list/[token]/page.tsx`（既存の `toAffiliateUrl` 関数）

優先順位:
1. リスト作成者の `affiliate_fanza_id`（`user_profiles` から取得）
2. システムデフォルト（環境変数 `NEXT_PUBLIC_FANZA_AFFILIATE_ID`）

```typescript
function toAffiliateUrl(raw?: string | null, curatorAffId?: string | null): string {
  const afId = curatorAffId ?? process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';
  // ... 既存のURL生成ロジックに afId を適用
}
```

FC2・MGSのアフィリエイトリンク生成も同様に対応する。

---

## 実装しないもの（MVP対象外）

- フォロー・フォロワー
- コメント・DM・通知
- ランキング
- ユーザー検索・キュレーター検索
- 一般ユーザー向けリスト作成UI改善
- いいね数のユニークユーザー集計

---

## 完了条件

- [ ] `/u/<username>` でプロフィールページが表示できる
- [ ] リスト詳細ページに作成者（アイコン+名前）が表示され、プロフィールへ遷移できる
- [ ] リスト閲覧数が表示・インクリメントされる
- [ ] リストいいねができ、カウントが表示される
- [ ] アカウント管理画面でアフィリエイトIDを設定できる
- [ ] リスト経由の外部リンクにキュレーターのアフィリエイトIDが適用される

---

## 実装順序（推奨）

1. DB マイグレーション（`user_profiles`, `view_count`, `list_likes`）
2. プロフィール編集UI + アカウント管理ページ拡張
3. リスト詳細ページへの作成者情報・閲覧数・いいね追加
4. 公開プロフィールページ（`/u/[username]`）
5. アフィリエイトリンク生成ロジック変更

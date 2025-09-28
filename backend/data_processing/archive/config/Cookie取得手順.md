# DMM サイト Cookie 取得手順

このガイドでは、DMMサイトで年齢認証を完了した後のCookieを取得する方法を説明します。

## 必要な作業

### 1. ブラウザでDMMサイトにアクセス

1. Google Chrome または Firefox を開く
2. https://www.dmm.co.jp/ にアクセス
3. 年齢認証ページが表示されたら「18歳以上」を選択して進む
4. サイトが正常に表示されることを確認

### 2. Cookie情報の取得

#### Chrome の場合：

1. **F12キーを押して開発者ツールを開く**
2. **「Application」タブをクリック**
3. **左側の「Storage」→「Cookies」を展開**
4. **「https://www.dmm.co.jp」または「https://review.dmm.co.jp」を選択**
5. **重要なCookieを確認**：
   - `ckcy` (重要)
   - `age_check_done` または類似の名前
   - セッション関連のCookie
   - その他認証関連のCookie

#### Firefox の場合：

1. **F12キーを押して開発者ツールを開く**
2. **「ストレージ」タブをクリック**
3. **「Cookie」→「https://www.dmm.co.jp」を展開**
4. **必要なCookieを確認**

### 3. Cookie データの保存

#### 方法1: JSON形式で保存（推奨）

`dmm_cookies.json` ファイルを作成して以下の形式で保存：

```json
{
  "ckcy": "1",
  "age_check_done": "1",
  "session_id": "取得したセッションID",
  "user_id": "取得したユーザーID"
}
```

#### 方法2: ブラウザ形式で保存

開発者ツールから「Copy all cookies」または手動で以下の形式でコピー：

```
ckcy=1; age_check_done=1; session_id=xxx; user_id=yyy
```

### 4. 重要なCookie一覧

DMMサイトで特に重要なCookieは以下の通りです：

| Cookie名 | 説明 | 重要度 |
|----------|------|--------|
| `ckcy` | 年齢認証確認 | ★★★ |
| `age_check_done` | 年齢チェック完了 | ★★★ |
| `session_id` | セッション識別 | ★★☆ |
| `login_token` | ログイントークン | ★★☆ |
| `user_pref` | ユーザー設定 | ★☆☆ |

## Cookie使用方法

### スクリプト実行前の準備

1. 上記で取得したCookieを `dmm_cookies.json` として保存
2. ファイルをスクリプトと同じディレクトリに配置

### スクリプトの実行

```bash
uv run python scripts/cookie_dmm_scraper.py
```

## トラブルシューティング

### Cookie認証に失敗する場合

1. **Cookieの期限切れ**
   - ブラウザで再度サイトにアクセス
   - 新しいCookieを取得

2. **必要なCookieが不足**
   - より多くのCookieを取得
   - 特に `ckcy` と年齢認証関連のCookieを確認

3. **サイトがまだ年齢認証を要求する場合**
   - ブラウザで完全に年齢認証を完了
   - プライベートブラウジングモードではないことを確認
   - JavaScriptが有効になっていることを確認

### Cookie取得時の注意点

- **プライバシー保護**: Cookieには個人情報が含まれる場合があります
- **期限**: Cookieには有効期限があります（通常数時間〜数日）
- **セキュリティ**: Cookieファイルは適切に管理してください

## Cookieの自動更新

より高度な使用法として、スクリプトが自動的にCookieを更新する仕組みも実装されています：

1. スクリプト実行時にCookieの有効性をテスト
2. 無効な場合は新しいCookieの取得を促す
3. 取得したCookieを自動保存

## サンプルCookieファイル

```json
{
  "ckcy": "1",
  "age_check_done": "1",
  "adult_verified": "true",
  "session_cookie": "abc123def456",
  "user_preferences": "adult_content_ok",
  "_ga": "GA1.2.xxxxx",
  "_gid": "GA1.2.yyyyy"
}
```

このファイルを参考に、実際に取得したCookie値を設定してください。
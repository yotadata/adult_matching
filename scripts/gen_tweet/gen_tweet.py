"""
動画からXツイート案を生成してDiscordに投稿するスクリプト

使い方:
  python gen_tweet.py [--count N] [--mode recent|popular]

環境変数:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  GEMINI_API_KEY
  DISCORD_WEBHOOK_URL
  FANZA_AFFILIATE_ID  (アフィリエイトリンク生成用)
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Optional

import httpx
from supabase import create_client

# ────────────────────────────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
FANZA_AFFILIATE_ID = os.environ.get("FANZA_AFFILIATE_ID", "")

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

SYSTEM_PROMPT = """\
あなたはアダルト動画のXアカウント運用担当者です。
以下の動画情報とサムネイル画像を見て、Xでバズりやすいツイート文を考えてください。

【ルール】
- 1ツイート目: 画像のみ投稿用の本文（URLなし、センシティブ指定なし）
  - 140文字以内
  - 絵文字を使って感情を込める（ｗｗ、！！など）
  - 「これはまずいでしょ」「大当たりすぎるｗ」のような話し言葉調
  - 内容・見た目・出演者の魅力を匂わせる（露骨な性的表現は禁止）
  - ハッシュタグは末尾に1〜2個（#AV #おすすめAV 系は避け、作品に合ったものを）
- 2ツイート目: 1ツイート目への返信として別アカウントから投稿するリンク誘導文
  - 60文字以内
  - FANZA購入リンクへ誘導
  - 例: 「フルはこちら👇 [AFFILIATE_URL]」「購入はここから→ [AFFILIATE_URL]」

出力はJSON形式で:
{
  "tweet1": "...",
  "tweet2": "...",
  "reason": "なぜこのコピーにしたか（30文字以内）"
}
"""


# ────────────────────────────────────────────────────────────────────
# DB から動画を取得
# ────────────────────────────────────────────────────────────────────
def fetch_videos(mode: str, count: int) -> list[dict]:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    query = sb.table("videos").select(
        "id, external_id, title, thumbnail_url, affiliate_url, product_url, "
        "performers:video_performers(performers(name)), "
        "tags:video_tags(tags(name))"
    ).not_.is_("thumbnail_url", "null")

    if mode == "popular":
        # like 数上位
        query = (
            sb.table("videos")
            .select(
                "id, external_id, title, thumbnail_url, affiliate_url, product_url, "
                "like_count, "
                "performers:video_performers(performers(name)), "
                "tags:video_tags(tags(name))"
            )
            .not_.is_("thumbnail_url", "null")
            .order("like_count", desc=True)
            .limit(count * 3)
        )
    else:
        # 直近追加
        query = (
            sb.table("videos")
            .select(
                "id, external_id, title, thumbnail_url, affiliate_url, product_url, "
                "performers:video_performers(performers(name)), "
                "tags:video_tags(tags(name))"
            )
            .not_.is_("thumbnail_url", "null")
            .order("created_at", desc=True)
            .limit(count * 3)
        )

    resp = query.execute()
    rows = resp.data or []

    # アフィリエイトリンクがあるものを優先
    has_affiliate = [r for r in rows if r.get("affiliate_url")]
    no_affiliate = [r for r in rows if not r.get("affiliate_url")]
    pool = has_affiliate + no_affiliate

    # ランダムに count 件
    selected = random.sample(pool, min(count, len(pool)))
    return selected


def build_affiliate_url(video: dict) -> str:
    if video.get("affiliate_url"):
        return video["affiliate_url"]
    if video.get("product_url"):
        return video["product_url"]
    # FANZA標準形式でフォールバック
    cid = video.get("external_id", "")
    if FANZA_AFFILIATE_ID and cid:
        return f"https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2Fvideoa%2F-%2Fdetail%2F%3D%2Fcid%3D{cid}%2F&af_id={FANZA_AFFILIATE_ID}&ch=link"
    return f"https://www.seihekilab.com/videos/{video['id']}"


def flatten_names(nested: list, key: str) -> list[str]:
    result = []
    for item in nested or []:
        inner = item.get(key)
        if isinstance(inner, dict):
            result.append(inner.get("name", ""))
        elif isinstance(inner, list):
            for x in inner:
                if isinstance(x, dict):
                    result.append(x.get("name", ""))
    return [n for n in result if n]


# ────────────────────────────────────────────────────────────────────
# Gemini API 呼び出し
# ────────────────────────────────────────────────────────────────────
def call_gemini(video: dict, affiliate_url: str) -> Optional[dict]:
    performers = flatten_names(video.get("performers", []), "performers")
    tags = flatten_names(video.get("tags", []), "tags")

    user_text = (
        f"タイトル: {video['title']}\n"
        f"出演者: {', '.join(performers) or '不明'}\n"
        f"ジャンル: {', '.join(tags[:10]) or '不明'}\n"
        f"アフィリエイトURL: {affiliate_url}\n"
    )

    parts = [{"text": SYSTEM_PROMPT + "\n\n" + user_text}]

    # サムネイル画像を URL で渡す
    thumbnail_url = video.get("thumbnail_url", "")
    if thumbnail_url:
        parts.append({
            "inline_data": None,  # placeholder; use fileData instead
        })
        # Gemini は URL 直接渡しは fileData / url フィールドで対応
        parts = [
            {"text": SYSTEM_PROMPT + "\n\n" + user_text},
            {"file_data": {"mime_type": "image/jpeg", "file_uri": thumbnail_url}},
        ]
        # file_uri が https URL の場合、Gemini 2.0 Flash は直接取得できる
        # ただし、DMM の画像は外部からアクセス可能なので問題なし

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.9,
            "maxOutputTokens": 512,
        },
    }

    try:
        resp = httpx.post(
            f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # JSON 抽出
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"[Gemini ERROR] {e}", file=sys.stderr)
    return None


# ────────────────────────────────────────────────────────────────────
# Discord 投稿
# ────────────────────────────────────────────────────────────────────
def post_to_discord(video: dict, affiliate_url: str, result: dict) -> None:
    performers = flatten_names(video.get("performers", []), "performers")
    title_short = video["title"][:50] + ("…" if len(video["title"]) > 50 else "")
    thumbnail = video.get("thumbnail_url", "")

    tweet1 = result.get("tweet1", "")
    tweet2 = result.get("tweet2", "")
    reason = result.get("reason", "")

    embeds = [
        {
            "title": f"🐦 ツイート案: {title_short}",
            "color": 0x1DA1F2,
            "fields": [
                {
                    "name": "【1ツイート目】メインアカウント（画像添付・URLなし）",
                    "value": f"```\n{tweet1}\n```",
                    "inline": False,
                },
                {
                    "name": "【2ツイート目】サブアカウント（返信・リンクあり）",
                    "value": f"```\n{tweet2}\n```",
                    "inline": False,
                },
                {
                    "name": "コピー意図",
                    "value": reason or "—",
                    "inline": True,
                },
                {
                    "name": "出演者",
                    "value": ", ".join(performers) or "不明",
                    "inline": True,
                },
                {
                    "name": "FANZA リンク",
                    "value": affiliate_url,
                    "inline": False,
                },
            ],
            "image": {"url": thumbnail} if thumbnail else None,
            "footer": {"text": f"video_id: {video['id']}"},
        }
    ]
    # None フィールド除去
    embeds[0] = {k: v for k, v in embeds[0].items() if v is not None}

    payload = {"embeds": embeds}
    resp = httpx.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    resp.raise_for_status()
    print(f"[Discord] posted: {title_short}")


# ────────────────────────────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="生成する動画数")
    parser.add_argument("--mode", choices=["recent", "popular"], default="recent")
    args = parser.parse_args()

    videos = fetch_videos(args.mode, args.count)
    if not videos:
        print("対象動画が見つかりませんでした")
        sys.exit(0)

    print(f"[gen_tweet] {len(videos)} 件処理します (mode={args.mode})")

    success = 0
    for video in videos:
        affiliate_url = build_affiliate_url(video)
        print(f"  → {video['title'][:40]}...")
        result = call_gemini(video, affiliate_url)
        if not result:
            print("    Gemini 失敗、スキップ")
            continue
        post_to_discord(video, affiliate_url, result)
        success += 1

    print(f"[gen_tweet] 完了: {success}/{len(videos)} 件投稿")


if __name__ == "__main__":
    main()

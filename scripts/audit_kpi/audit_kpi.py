#!/usr/bin/env python3
"""
推薦品質 KPI 監視スクリプト

直近 N 日間の user_video_decisions から KPI を集計し、
閾値を超えた場合に Discord に通知する。

使用例:
  python audit_kpi.py --db-url postgresql://... --discord-url https://discord.com/api/webhooks/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

SITE_URL = "https://seihekilab.com"

METRICS_QUERY = """
select
    count(*) filter (where decision_type = 'like')  as likes,
    count(*) filter (where decision_type = 'nope')  as nopes,
    count(*)                                          as total_decisions,
    count(distinct user_id)                           as active_users,
    count(*) filter (where recommendation_type is not null) as recommended_decisions,
    count(*) filter (
        where decision_type = 'like'
          and recommendation_type is not null
    ) as recommended_likes
from public.user_video_decisions
where created_at >= now() - interval '{days} days'
"""

DAILY_QUERY = """
select
    date_trunc('day', created_at at time zone 'Asia/Tokyo')::date as day,
    count(*) filter (where decision_type = 'like')  as likes,
    count(*) filter (where decision_type = 'nope')  as nopes,
    count(distinct user_id)                           as dau
from public.user_video_decisions
where created_at >= now() - interval '{days} days'
group by 1
order by 1 desc
limit 7
"""

SOURCE_LIKE_RATE_QUERY = """
select
    recommendation_source,
    count(*) filter (where decision_type = 'like') as likes,
    count(*)                                        as total
from public.user_video_decisions
where created_at >= now() - interval '{days} days'
  and recommendation_source is not null
group by 1
order by 1
"""

SCORE_CALIBRATION_QUERY = """
select
    case
        when recommendation_score < 0.3 then '低 (0.0-0.3)'
        when recommendation_score < 0.6 then '中 (0.3-0.6)'
        else                                  '高 (0.6-1.0)'
    end as score_bucket,
    count(*) filter (where decision_type = 'like') as likes,
    count(*)                                        as total
from public.user_video_decisions
where created_at >= now() - interval '{days} days'
  and recommendation_score is not null
group by 1
order by 1
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit recommendation KPIs")
    p.add_argument("--db-url", default=os.environ.get("REMOTE_DATABASE_URL", ""))
    p.add_argument("--discord-url", default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    p.add_argument("--window-days", type=int, default=7, help="集計対象の日数")
    p.add_argument("--min-like-rate", type=float, default=0.30, help="like率の最低閾値（下回ったら警告）")
    p.add_argument("--min-dau", type=int, default=5, help="DAU の最低閾値")
    p.add_argument("--output-json", default="", help="結果を JSON に出力するパス")
    return p.parse_args()


def connect(db_url: str):
    return psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_metrics(conn, window_days: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(METRICS_QUERY.format(days=window_days))
        row = dict(cur.fetchone() or {})

    with conn.cursor() as cur:
        cur.execute(DAILY_QUERY.format(days=window_days))
        daily = [dict(r) for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute(SOURCE_LIKE_RATE_QUERY.format(days=window_days))
        source_rows = [dict(r) for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute(SCORE_CALIBRATION_QUERY.format(days=window_days))
        score_rows = [dict(r) for r in cur.fetchall()]

    likes = int(row.get("likes") or 0)
    nopes = int(row.get("nopes") or 0)
    total = int(row.get("total_decisions") or 0)
    like_rate = likes / total if total > 0 else 0.0

    rec_total = int(row.get("recommended_decisions") or 0)
    rec_likes = int(row.get("recommended_likes") or 0)
    rec_like_rate = rec_likes / rec_total if rec_total > 0 else 0.0

    avg_dau = sum(int(d.get("dau") or 0) for d in daily) / len(daily) if daily else 0.0

    source_like_rates = {
        r["recommendation_source"]: {
            "likes": int(r["likes"] or 0),
            "total": int(r["total"] or 0),
            "like_rate": round(int(r["likes"] or 0) / int(r["total"]) if int(r["total"]) > 0 else 0.0, 4),
        }
        for r in source_rows
    }

    score_calibration = {
        r["score_bucket"]: {
            "likes": int(r["likes"] or 0),
            "total": int(r["total"] or 0),
            "like_rate": round(int(r["likes"] or 0) / int(r["total"]) if int(r["total"]) > 0 else 0.0, 4),
        }
        for r in score_rows
    }

    return {
        "window_days": window_days,
        "likes": likes,
        "nopes": nopes,
        "total_decisions": total,
        "like_rate": round(like_rate, 4),
        "active_users": int(row.get("active_users") or 0),
        "recommended_like_rate": round(rec_like_rate, 4),
        "avg_dau": round(avg_dau, 1),
        "source_like_rates": source_like_rates,
        "score_calibration": score_calibration,
        "daily": [
            {
                "day": str(d["day"]),
                "likes": int(d.get("likes") or 0),
                "nopes": int(d.get("nopes") or 0),
                "dau": int(d.get("dau") or 0),
            }
            for d in daily
        ],
    }


def _fmt_source_lines(source_like_rates: dict) -> str:
    source_labels = {
        "exploitation": "個人推薦",
        "popularity":   "人気",
        "exploration":  "探索",
    }
    lines = []
    for src, label in source_labels.items():
        if src not in source_like_rates:
            continue
        d = source_like_rates[src]
        lines.append(f"{label}: {d['like_rate']:.1%} ({d['likes']}/{d['total']}件)")
    return "\n".join(lines) or "データなし"


def _fmt_score_lines(score_calibration: dict) -> str:
    lines = []
    for bucket in ["低 (0.0-0.3)", "中 (0.3-0.6)", "高 (0.6-1.0)"]:
        if bucket not in score_calibration:
            continue
        d = score_calibration[bucket]
        lines.append(f"{bucket}: {d['like_rate']:.1%} ({d['total']}件)")
    return "\n".join(lines) or "データなし"


def build_discord_message(metrics: dict, alerts: list[str], args: argparse.Namespace) -> dict:
    status = "🚨 警告あり" if alerts else "✅ 正常"
    like_rate_pct = f"{metrics['like_rate']:.1%}"
    rec_like_rate_pct = f"{metrics['recommended_like_rate']:.1%}"

    daily_lines = []
    for d in metrics["daily"][:5]:
        total = d["likes"] + d["nopes"]
        rate = f"{d['likes']/total:.0%}" if total > 0 else "-"
        daily_lines.append(f"`{d['day']}` DAU={d['dau']} like率={rate}")

    alert_text = "\n".join(f"⚠️ {a}" for a in alerts) if alerts else "なし"

    embed = {
        "title": f"[KPI監視] {status} — 直近{metrics['window_days']}日",
        "color": 0xFF4444 if alerts else 0x44AA44,
        "fields": [
            {"name": "📊 全体 like 率", "value": like_rate_pct, "inline": True},
            {"name": "🤖 推薦 like 率", "value": rec_like_rate_pct, "inline": True},
            {"name": "👥 平均 DAU", "value": str(metrics["avg_dau"]), "inline": True},
            {"name": "🎯 ソース別 like 率", "value": _fmt_source_lines(metrics["source_like_rates"]), "inline": False},
            {"name": "📈 スコアキャリブレーション", "value": _fmt_score_lines(metrics["score_calibration"]), "inline": False},
            {"name": "📅 日別（直近5日）", "value": "\n".join(daily_lines) or "データなし", "inline": False},
            {"name": "⚠️ アラート", "value": alert_text, "inline": False},
        ],
        "footer": {"text": f"measured_at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"},
        "url": SITE_URL,
    }
    return {"embeds": [embed]}


def send_discord(webhook_url: str, payload: dict) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"Discord webhook returned {resp.status}")


def main() -> None:
    args = parse_args()

    if not args.db_url:
        print("[error] --db-url または REMOTE_DATABASE_URL が未設定", file=sys.stderr)
        sys.exit(1)

    print(f"[kpi] DB 接続中...")
    conn = connect(args.db_url)

    print(f"[kpi] 直近 {args.window_days} 日間の KPI を集計中...")
    metrics = fetch_metrics(conn, args.window_days)
    conn.close()

    alerts: list[str] = []
    if metrics["total_decisions"] > 0 and metrics["like_rate"] < args.min_like_rate:
        alerts.append(
            f"like率が低下: {metrics['like_rate']:.1%} < 閾値 {args.min_like_rate:.1%}"
        )
    if metrics["avg_dau"] < args.min_dau and metrics["total_decisions"] > 0:
        alerts.append(
            f"DAU 低下: avg={metrics['avg_dau']} < 閾値 {args.min_dau}"
        )

    print(f"\n--- KPI サマリー (直近{metrics['window_days']}日) ---")
    print(f"  like率:      {metrics['like_rate']:.1%}")
    print(f"  推薦like率:  {metrics['recommended_like_rate']:.1%}")
    print(f"  平均DAU:     {metrics['avg_dau']}")
    print(f"  総判断数:    {metrics['total_decisions']:,}")

    print("\n  [ソース別 like 率]")
    source_labels = {"exploitation": "個人推薦", "popularity": "人気", "exploration": "探索"}
    for src, label in source_labels.items():
        d = metrics["source_like_rates"].get(src)
        if d:
            print(f"    {label}: {d['like_rate']:.1%} ({d['likes']}/{d['total']}件)")
        else:
            print(f"    {label}: データなし")

    print("\n  [スコアキャリブレーション]")
    for bucket in ["低 (0.0-0.3)", "中 (0.3-0.6)", "高 (0.6-1.0)"]:
        d = metrics["score_calibration"].get(bucket)
        if d:
            print(f"    {bucket}: {d['like_rate']:.1%} ({d['total']}件)")
        else:
            print(f"    {bucket}: データなし")

    if alerts:
        print("\n⚠️ アラート:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("\n✅ 全指標が閾値内")

    if args.output_json:
        out = {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "alerts": alerts,
            "metrics": metrics,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        print(f"[kpi] JSON 出力: {args.output_json}")

    if args.discord_url:
        payload = build_discord_message(metrics, alerts, args)
        send_discord(args.discord_url, payload)
        print("[kpi] Discord 通知送信完了")
    elif alerts:
        print("[kpi] DISCORD_WEBHOOK_URL 未設定のため通知スキップ")

    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()

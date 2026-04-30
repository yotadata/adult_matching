#!/usr/bin/env python3
"""
maker 偏り監視スクリプト

user_video_decisions の推薦履歴から maker 別分布を集計し、
偏りが閾値を超えた場合に警告する。結果は CSV に追記する。

使用例:
  python audit_maker_bias.py --db-url postgresql://... --log-csv docs/ml/maker_bias_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras


QUERY = """
select
    v.maker,
    count(*) as recommendations,
    count(distinct uvd.video_id) as unique_items,
    count(distinct uvd.user_id) as user_coverage
from public.user_video_decisions uvd
join public.videos v on v.id = uvd.video_id
where uvd.recommendation_type is not null
  and uvd.created_at >= now() - interval '{days} days'
group by v.maker
order by recommendations desc
"""

TOTAL_QUERY = """
select count(*) as total
from public.user_video_decisions uvd
where uvd.recommendation_type is not null
  and uvd.created_at >= now() - interval '{days} days'
"""

LOG_HEADER = [
    "measured_at",
    "window_days",
    "maker",
    "recommendations",
    "unique_items",
    "user_coverage",
    "share",
    "bias_flagged",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit maker bias in recommendations")
    p.add_argument("--db-url", default=os.environ.get("REMOTE_DATABASE_URL", ""))
    p.add_argument("--window-days", type=int, default=30, help="集計対象の日数")
    p.add_argument("--threshold", type=float, default=0.20, help="偏り警告閾値（share）")
    p.add_argument("--log-csv", default="docs/ml/maker_bias_log.csv", help="追記先 CSV")
    p.add_argument("--summary-json", default="", help="結果を JSON に出力する場合のパス")
    p.add_argument("--top-n", type=int, default=20, help="表示する上位 maker 数")
    return p.parse_args()


def connect(db_url: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_distribution(conn, window_days: int) -> tuple[pd.DataFrame, int]:
    with conn.cursor() as cur:
        cur.execute(TOTAL_QUERY.format(days=window_days))
        total = int((cur.fetchone() or {}).get("total", 0))

    with conn.cursor() as cur:
        cur.execute(QUERY.format(days=window_days))
        rows = cur.fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df, total

    df["recommendations"] = df["recommendations"].astype(int)
    df["unique_items"] = df["unique_items"].astype(int)
    df["user_coverage"] = df["user_coverage"].astype(int)
    df["share"] = df["recommendations"] / total if total > 0 else 0.0
    return df, total


def append_log(log_path: Path, rows: list[dict], measured_at: str) -> None:
    exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.db_url:
        print("[error] --db-url または REMOTE_DATABASE_URL が未設定", file=sys.stderr)
        sys.exit(1)

    print(f"[audit] DB 接続中...")
    conn = connect(args.db_url)

    print(f"[audit] 直近 {args.window_days} 日間の推薦分布を集計中...")
    df, total = fetch_distribution(conn, args.window_days)
    conn.close()

    if df.empty or total == 0:
        print(f"[audit] 推薦データなし（window={args.window_days}日）。スキップ。")
        return

    print(f"\n[audit] 総推薦数: {total:,}  window: {args.window_days}日  閾値: {args.threshold:.0%}")
    print(f"\n{'maker':<30} {'推薦数':>8} {'ユニーク':>8} {'ユーザー':>8} {'share':>7} {'⚠️ 偏り'}")
    print("-" * 75)

    flagged = []
    measured_at = datetime.now(timezone.utc).isoformat()
    log_rows = []
    bias_detected = False

    for _, row in df.head(args.top_n).iterrows():
        flag = row["share"] >= args.threshold
        if flag:
            flagged.append(row["maker"])
            bias_detected = True
        marker = "⚠️ " if flag else "   "
        print(
            f"{str(row['maker']):<30} {int(row['recommendations']):>8,} "
            f"{int(row['unique_items']):>8,} {int(row['user_coverage']):>8,} "
            f"{row['share']:>6.1%}  {marker}"
        )
        log_rows.append({
            "measured_at": measured_at,
            "window_days": args.window_days,
            "maker": row["maker"],
            "recommendations": int(row["recommendations"]),
            "unique_items": int(row["unique_items"]),
            "user_coverage": int(row["user_coverage"]),
            "share": round(float(row["share"]), 6),
            "bias_flagged": flag,
        })

    if flagged:
        print(f"\n[audit] ⚠️  偏り検出: {', '.join(flagged)} が閾値 {args.threshold:.0%} を超えています。")
    else:
        print(f"\n[audit] ✅ 偏りなし（全 maker が閾値 {args.threshold:.0%} 以下）")

    log_path = Path(args.log_csv)
    append_log(log_path, log_rows, measured_at)
    print(f"[audit] CSV に追記: {log_path}  ({len(log_rows)} 行)")

    if args.summary_json:
        summary = {
            "measured_at": measured_at,
            "window_days": args.window_days,
            "total_recommendations": total,
            "threshold": args.threshold,
            "bias_detected": bias_detected,
            "flagged_makers": flagged,
            "top_makers": df.head(args.top_n).to_dict(orient="records"),
        }
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[audit] JSON 出力: {args.summary_json}")

    if bias_detected:
        sys.exit(2)


if __name__ == "__main__":
    main()

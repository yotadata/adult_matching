"""
スクレイピング進捗監視ツール
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys

class ProgressMonitor:
    def __init__(self):
        self.progress_file = Path("../raw_data/scraping_progress.json")
        self.log_file = Path("../raw_data/scraping.log")
        self.pid_file = Path("../raw_data/scraper.pid")
        self.output_dir = Path("../raw_data/batch_reviews")
    
    def is_running(self) -> bool:
        """スクレイパーが実行中かチェック"""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # プロセスが存在するかチェック
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        except:
            return False
    
    def load_progress(self) -> dict:
        """進捗データを読み込み"""
        try:
            if not self.progress_file.exists():
                return {}
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def count_output_files(self) -> int:
        """出力ファイル数をカウント"""
        if not self.output_dir.exists():
            return 0
        
        return len([f for f in self.output_dir.glob("reviewer_*.json")])
    
    def get_file_stats(self) -> dict:
        """ファイル統計を取得"""
        if not self.output_dir.exists():
            return {"total_files": 0, "total_size": 0}
        
        total_size = 0
        total_files = 0
        
        for file_path in self.output_dir.glob("reviewer_*.json"):
            total_files += 1
            total_size += file_path.stat().st_size
        
        return {"total_files": total_files, "total_size": total_size}
    
    def format_size(self, size_bytes: int) -> str:
        """サイズをフォーマット"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    
    def format_duration(self, start_time_str: str) -> str:
        """実行時間をフォーマット"""
        try:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed = datetime.now() - start_time
            
            hours = elapsed.seconds // 3600
            minutes = (elapsed.seconds % 3600) // 60
            seconds = elapsed.seconds % 60
            
            if elapsed.days > 0:
                return f"{elapsed.days}日 {hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "不明"
    
    def show_status(self):
        """現在の状況を表示"""
        print("=== スクレイピング進捗状況 ===")
        print()
        
        # 実行状況
        is_running = self.is_running()
        print(f"実行状況: {'🟢 実行中' if is_running else '🔴 停止中'}")
        
        # 進捗データ読み込み
        progress = self.load_progress()
        
        if not progress:
            print("進捗データが見つかりません")
            return
        
        # 基本情報
        print(f"セッションID: {progress.get('session_id', 'N/A')}")
        
        if progress.get('start_time'):
            print(f"開始時刻: {progress['start_time']}")
            print(f"実行時間: {self.format_duration(progress['start_time'])}")
        
        print()
        
        # 進捗情報
        total_reviewers = progress.get('total_reviewers', 0)
        completed = len(progress.get('completed_reviewers', []))
        failed = len(progress.get('failed_reviewers', []))
        current = progress.get('current_reviewer', 'なし')
        
        print("=== 進捗詳細 ===")
        print(f"対象レビュワー総数: {total_reviewers}")
        print(f"完了済み: {completed}")
        print(f"失敗: {failed}")
        print(f"残り: {total_reviewers - completed - failed}")
        
        if total_reviewers > 0:
            progress_pct = (completed / total_reviewers) * 100
            print(f"完了率: {progress_pct:.1f}%")
        
        print(f"現在処理中: {current}")
        print()
        
        # ファイル統計
        file_stats = self.get_file_stats()
        print("=== ファイル統計 ===")
        print(f"出力ファイル数: {file_stats['total_files']}")
        print(f"総データサイズ: {self.format_size(file_stats['total_size'])}")
        
        total_reviews = progress.get('total_reviews_collected', 0)
        print(f"収集レビュー数: {total_reviews:,}")
        
        if completed > 0:
            avg_reviews = total_reviews / completed
            print(f"平均レビュー数/人: {avg_reviews:.1f}")
        
        print()
        
        # 推定残り時間
        if is_running and completed > 0 and progress.get('start_time'):
            try:
                start_time = datetime.fromisoformat(progress['start_time'])
                elapsed = datetime.now() - start_time
                
                remaining = total_reviewers - completed - failed
                if remaining > 0:
                    avg_time_per_reviewer = elapsed.total_seconds() / completed
                    estimated_remaining = remaining * avg_time_per_reviewer
                    
                    estimated_finish = datetime.now() + timedelta(seconds=estimated_remaining)
                    
                    print("=== 推定 ===")
                    print(f"残り時間: {self.format_duration(datetime.now().isoformat())} ※参考値")
                    print(f"完了予定: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')} ※参考値")
                    print()
            except:
                pass
        
        # 最近の完了レビュワー
        if progress.get('completed_reviewers'):
            recent_completed = progress['completed_reviewers'][-5:]
            print("=== 最近完了したレビュワー（最新5人） ===")
            for reviewer_id in recent_completed:
                print(f"  - {reviewer_id}")
            print()
        
        # 失敗レビュワー
        if progress.get('failed_reviewers'):
            print("=== 失敗レビュワー ===")
            for reviewer_id in progress['failed_reviewers']:
                print(f"  - {reviewer_id}")
            print()
    
    def tail_log(self, lines: int = 20):
        """ログの最後の部分を表示"""
        try:
            if not self.log_file.exists():
                print("ログファイルが見つかりません")
                return
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            recent_lines = all_lines[-lines:]
            
            print(f"=== ログ最新{lines}行 ===")
            for line in recent_lines:
                print(line.rstrip())
            
        except Exception as e:
            print(f"ログ読み込みエラー: {e}")
    
    def watch_progress(self, interval: int = 30):
        """進捗を定期的に監視"""
        print(f"進捗監視開始（{interval}秒間隔）")
        print("Ctrl+C で停止")
        print()
        
        try:
            while True:
                # クリア
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # 状況表示
                self.show_status()
                
                # 実行中でなければ終了
                if not self.is_running():
                    print("スクレイパーが停止しています")
                    break
                
                print(f"次回更新まで {interval} 秒...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n監視を終了しました")

def main():
    parser = argparse.ArgumentParser(description='スクレイピング進捗監視')
    parser.add_argument('--watch', action='store_true', help='連続監視モード')
    parser.add_argument('--interval', type=int, default=30, help='監視間隔（秒）')
    parser.add_argument('--log', type=int, help='ログの最新N行を表示')
    
    args = parser.parse_args()
    
    monitor = ProgressMonitor()
    
    if args.log:
        monitor.tail_log(args.log)
    elif args.watch:
        monitor.watch_progress(args.interval)
    else:
        monitor.show_status()

if __name__ == "__main__":
    main()
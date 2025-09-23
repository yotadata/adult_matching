"""
DMM Sync Monitor

DMM同期操作の監視・メトリクス・アラート管理
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SyncMetrics:
    """同期メトリクス"""
    sync_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_items: int = 0
    new_items: int = 0
    updated_items: int = 0
    skipped_items: int = 0
    error_count: int = 0
    api_calls: int = 0
    rate_limit_hits: int = 0
    success: bool = False
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def items_per_second(self) -> Optional[float]:
        if self.duration_seconds and self.duration_seconds > 0:
            return self.total_items / self.duration_seconds
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        # datetimeを文字列に変換
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        data['duration_seconds'] = self.duration_seconds
        data['items_per_second'] = self.items_per_second
        return data


class SyncMonitor:
    """DMM同期監視システム"""
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        self.metrics_dir = metrics_dir or Path("backend/data/storage/logs/sync")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics: Optional[SyncMetrics] = None
        
    def start_sync(self, sync_id: Optional[str] = None) -> str:
        """
        同期の開始監視
        
        Args:
            sync_id: 同期ID（Noneで自動生成）
            
        Returns:
            同期ID
        """
        if not sync_id:
            sync_id = f"dmm_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_metrics = SyncMetrics(
            sync_id=sync_id,
            start_time=datetime.utcnow()
        )
        
        logger.info(f"🚀 Started monitoring sync: {sync_id}")
        return sync_id
    
    def update_progress(
        self,
        total_items: int = 0,
        new_items: int = 0, 
        updated_items: int = 0,
        skipped_items: int = 0,
        api_calls: int = 0,
        rate_limit_hits: int = 0
    ):
        """進捗更新"""
        if not self.current_metrics:
            return
            
        self.current_metrics.total_items += total_items
        self.current_metrics.new_items += new_items
        self.current_metrics.updated_items += updated_items
        self.current_metrics.skipped_items += skipped_items
        self.current_metrics.api_calls += api_calls
        self.current_metrics.rate_limit_hits += rate_limit_hits
    
    def record_error(self, error: Exception, context: str = ""):
        """エラー記録"""
        if not self.current_metrics:
            return
            
        self.current_metrics.error_count += 1
        logger.error(f"Sync error in {context}: {error}")
        
        # エラーしきい値チェック
        if self.current_metrics.error_count >= 10:
            logger.critical(f"High error count ({self.current_metrics.error_count}) in sync {self.current_metrics.sync_id}")
    
    def finish_sync(self, success: bool = True) -> SyncMetrics:
        """
        同期完了監視
        
        Args:
            success: 同期成功フラグ
            
        Returns:
            最終メトリクス
        """
        if not self.current_metrics:
            raise ValueError("No active sync to finish")
            
        self.current_metrics.end_time = datetime.utcnow()
        self.current_metrics.success = success
        
        # メトリクス保存
        self._save_metrics(self.current_metrics)
        
        # サマリーログ
        logger.info(f"🏁 Finished sync {self.current_metrics.sync_id}:")
        logger.info(f"   Success: {success}")
        logger.info(f"   Duration: {self.current_metrics.duration_seconds:.2f}s")
        logger.info(f"   Items: {self.current_metrics.total_items} ({self.current_metrics.items_per_second:.2f}/s)")
        logger.info(f"   New: {self.current_metrics.new_items}, Updated: {self.current_metrics.updated_items}")
        logger.info(f"   Errors: {self.current_metrics.error_count}")
        
        result = self.current_metrics
        self.current_metrics = None
        return result
    
    def _save_metrics(self, metrics: SyncMetrics):
        """メトリクスをファイルに保存"""
        try:
            metrics_file = self.metrics_dir / f"{metrics.sync_id}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_recent_metrics(self, days: int = 7) -> List[SyncMetrics]:
        """
        最近のメトリクス取得
        
        Args:
            days: 取得日数
            
        Returns:
            メトリクスリスト
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        metrics_list = []
        
        try:
            for metrics_file in self.metrics_dir.glob("*_metrics.json"):
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    start_time = datetime.fromisoformat(data['start_time'])
                    if start_time >= cutoff_date:
                        # 辞書からSyncMetricsに変換
                        metrics = SyncMetrics(
                            sync_id=data['sync_id'],
                            start_time=start_time,
                            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                            total_items=data.get('total_items', 0),
                            new_items=data.get('new_items', 0),
                            updated_items=data.get('updated_items', 0),
                            skipped_items=data.get('skipped_items', 0),
                            error_count=data.get('error_count', 0),
                            api_calls=data.get('api_calls', 0),
                            rate_limit_hits=data.get('rate_limit_hits', 0),
                            success=data.get('success', False)
                        )
                        metrics_list.append(metrics)
                except Exception as e:
                    logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to read metrics directory: {e}")
        
        return sorted(metrics_list, key=lambda x: x.start_time, reverse=True)
    
    def generate_sync_report(self, days: int = 30) -> Dict[str, Any]:
        """
        同期レポート生成
        
        Args:
            days: レポート期間（日数）
            
        Returns:
            同期レポート
        """
        metrics_list = self.get_recent_metrics(days)
        
        if not metrics_list:
            return {"error": "No sync data available"}
        
        # 統計計算
        total_syncs = len(metrics_list)
        successful_syncs = len([m for m in metrics_list if m.success])
        success_rate = (successful_syncs / total_syncs) * 100 if total_syncs > 0 else 0
        
        total_items_processed = sum(m.total_items for m in metrics_list)
        total_new_items = sum(m.new_items for m in metrics_list)
        total_updated_items = sum(m.updated_items for m in metrics_list)
        total_errors = sum(m.error_count for m in metrics_list)
        
        durations = [m.duration_seconds for m in metrics_list if m.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        throughputs = [m.items_per_second for m in metrics_list if m.items_per_second]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        
        return {
            "report_period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_syncs": total_syncs,
                "successful_syncs": successful_syncs,
                "success_rate_percent": round(success_rate, 2),
                "total_items_processed": total_items_processed,
                "total_new_items": total_new_items,
                "total_updated_items": total_updated_items,
                "total_errors": total_errors,
                "average_duration_seconds": round(avg_duration, 2),
                "average_throughput_items_per_second": round(avg_throughput, 2)
            },
            "recent_syncs": [
                {
                    "sync_id": m.sync_id,
                    "start_time": m.start_time.isoformat(),
                    "success": m.success,
                    "duration_seconds": m.duration_seconds,
                    "total_items": m.total_items,
                    "error_count": m.error_count
                }
                for m in metrics_list[:10]  # 最新10件
            ]
        }


# シングルトンインスタンス
_monitor_instance = None

def get_sync_monitor() -> SyncMonitor:
    """同期監視インスタンス取得"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SyncMonitor()
    return _monitor_instance
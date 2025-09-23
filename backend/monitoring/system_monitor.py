"""
System Monitor for Adult Matching Backend

リファクタリングされたバックエンドシステムの包括的監視
- メトリクス収集とエクスポート
- ヘルスチェックと状態監視
- アラート統合とログ管理
"""

import asyncio
import time
import psutil
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
import threading

# Prometheus メトリクス定義
REQUEST_COUNT = Counter('adult_matching_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('adult_matching_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
SYSTEM_CPU_USAGE = Gauge('adult_matching_system_cpu_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('adult_matching_system_memory_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('adult_matching_system_disk_percent', 'System disk usage percentage')
ML_MODEL_ACCURACY = Gauge('adult_matching_ml_model_accuracy', 'ML model accuracy score')
DATA_QUALITY_SCORE = Gauge('adult_matching_data_quality_score', 'Data quality assessment score')
USER_ACTIVITY_RATE = Gauge('adult_matching_user_activity_rate', 'User activity rate per hour')
RECOMMENDATION_LATENCY = Histogram('adult_matching_recommendation_latency_seconds', 'Recommendation response time')
DATABASE_CONNECTIONS = Gauge('adult_matching_db_connections', 'Active database connections')

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """システムヘルス状態"""
    timestamp: datetime
    status: str  # healthy, warning, critical
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    response_time_ms: float
    error_rate_percent: float
    uptime_seconds: int
    
@dataclass
class ServiceMetrics:
    """サービスメトリクス"""
    service_name: str
    timestamp: datetime
    status: str
    request_rate: float
    error_rate: float
    avg_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float

class SystemMonitor:
    """統合システム監視"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.start_time = time.time()
        self.health_history: List[SystemHealth] = []
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.monitoring_enabled = True
        
        # Prometheus HTTP server
        self.prometheus_port = self.config.get('prometheus_port', 8000)
        
        logger.info(f"System monitor initialized with config: {self.config}")
    
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'collection_interval': 15,  # seconds
            'alert_thresholds': {
                'cpu_critical': 90.0,
                'cpu_warning': 75.0,
                'memory_critical': 95.0,
                'memory_warning': 80.0,
                'disk_critical': 95.0,
                'disk_warning': 85.0,
                'response_time_critical': 5000,  # ms
                'response_time_warning': 2000,   # ms
                'error_rate_critical': 10.0,    # %
                'error_rate_warning': 5.0       # %
            },
            'prometheus_port': 8000,
            'health_check_endpoints': [
                'http://localhost:54321/health',
                'http://localhost:3000/api/health'
            ],
            'services_to_monitor': [
                'enhanced_two_tower_recommendations',
                'user-management',
                'content-feed',
                'data-sync',
                'ml-training'
            ]
        }
    
    async def start_monitoring(self):
        """監視開始"""
        logger.info("Starting system monitoring...")
        
        # Prometheus HTTPサーバー開始
        self._start_prometheus_server()
        
        # 監視タスクの開始
        monitoring_tasks = [
            self._monitor_system_resources(),
            self._monitor_services(),
            self._monitor_health_endpoints(),
            self._collect_business_metrics(),
            self._process_alerts()
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    def _start_prometheus_server(self):
        """Prometheus HTTPサーバー開始"""
        def start_server():
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        # 別スレッドでPrometheusサーバーを起動
        prometheus_thread = threading.Thread(target=start_server, daemon=True)
        prometheus_thread.start()
    
    async def _monitor_system_resources(self):
        """システムリソース監視"""
        while self.monitoring_enabled:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                SYSTEM_CPU_USAGE.set(cpu_percent)
                
                # メモリ使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                SYSTEM_MEMORY_USAGE.set(memory_percent)
                
                # ディスク使用率
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                SYSTEM_DISK_USAGE.set(disk_percent)
                
                # データベース接続数
                db_connections = await self._get_db_connections()
                DATABASE_CONNECTIONS.set(db_connections)
                
                # ヘルス状態の評価
                health_status = self._evaluate_health_status(
                    cpu_percent, memory_percent, disk_percent
                )
                
                # ヘルス履歴に追加
                health = SystemHealth(
                    timestamp=datetime.now(),
                    status=health_status,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_percent=disk_percent,
                    active_connections=db_connections,
                    response_time_ms=0,  # 後で更新
                    error_rate_percent=0,  # 後で更新
                    uptime_seconds=int(time.time() - self.start_time)
                )
                
                self.health_history.append(health)
                
                # 履歴サイズ制限
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-500:]
                
                logger.debug(f"System metrics: CPU={cpu_percent}%, Memory={memory_percent}%, Disk={disk_percent}%")
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
            
            await asyncio.sleep(self.config['collection_interval'])
    
    async def _monitor_services(self):
        """サービス監視"""
        while self.monitoring_enabled:
            try:
                for service in self.config['services_to_monitor']:
                    metrics = await self._collect_service_metrics(service)
                    if metrics:
                        self.service_metrics[service] = metrics
                        
                        # Prometheusメトリクスの更新
                        REQUEST_COUNT.labels(
                            method='GET', 
                            endpoint=service, 
                            status='200'
                        ).inc(metrics.request_rate)
                        
                        REQUEST_DURATION.labels(
                            method='GET', 
                            endpoint=service
                        ).observe(metrics.avg_response_time / 1000)  # ms to seconds
                
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
            
            await asyncio.sleep(self.config['collection_interval'] * 2)
    
    async def _monitor_health_endpoints(self):
        """ヘルスエンドポイント監視"""
        while self.monitoring_enabled:
            try:
                async with aiohttp.ClientSession() as session:
                    for endpoint in self.config['health_check_endpoints']:
                        start_time = time.time()
                        try:
                            async with session.get(endpoint, timeout=5) as response:
                                response_time = (time.time() - start_time) * 1000
                                
                                if response.status == 200:
                                    logger.debug(f"Health check OK: {endpoint} ({response_time:.2f}ms)")
                                else:
                                    logger.warning(f"Health check failed: {endpoint} (status={response.status})")
                                    
                                # Prometheusメトリクス更新
                                REQUEST_COUNT.labels(
                                    method='GET',
                                    endpoint='health',
                                    status=str(response.status)
                                ).inc()
                                
                                REQUEST_DURATION.labels(
                                    method='GET',
                                    endpoint='health'
                                ).observe(response_time / 1000)
                        
                        except asyncio.TimeoutError:
                            logger.error(f"Health check timeout: {endpoint}")
                        except Exception as e:
                            logger.error(f"Health check error for {endpoint}: {e}")
            
            except Exception as e:
                logger.error(f"Error monitoring health endpoints: {e}")
            
            await asyncio.sleep(30)  # ヘルスチェックは30秒間隔
    
    async def _collect_business_metrics(self):
        """ビジネスメトリクス収集"""
        while self.monitoring_enabled:
            try:
                # MLモデル精度の取得
                model_accuracy = await self._get_model_accuracy()
                if model_accuracy is not None:
                    ML_MODEL_ACCURACY.set(model_accuracy)
                
                # データ品質スコアの取得
                data_quality = await self._get_data_quality_score()
                if data_quality is not None:
                    DATA_QUALITY_SCORE.set(data_quality)
                
                # ユーザーアクティビティレートの取得
                user_activity = await self._get_user_activity_rate()
                if user_activity is not None:
                    USER_ACTIVITY_RATE.set(user_activity)
                
                logger.debug(f"Business metrics: accuracy={model_accuracy}, quality={data_quality}, activity={user_activity}")
            
            except Exception as e:
                logger.error(f"Error collecting business metrics: {e}")
            
            await asyncio.sleep(self.config['collection_interval'] * 4)  # 1分間隔
    
    async def _process_alerts(self):
        """アラート処理"""
        while self.monitoring_enabled:
            try:
                if self.health_history:
                    latest_health = self.health_history[-1]
                    alerts = self._evaluate_alerts(latest_health)
                    
                    for alert in alerts:
                        await self._send_alert(alert)
            
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
            
            await asyncio.sleep(60)  # アラート処理は1分間隔
    
    def _evaluate_health_status(self, cpu_percent: float, memory_percent: float, disk_percent: float) -> str:
        """ヘルス状態評価"""
        thresholds = self.alert_thresholds
        
        if (cpu_percent > thresholds['cpu_critical'] or 
            memory_percent > thresholds['memory_critical'] or 
            disk_percent > thresholds['disk_critical']):
            return 'critical'
        
        if (cpu_percent > thresholds['cpu_warning'] or 
            memory_percent > thresholds['memory_warning'] or 
            disk_percent > thresholds['disk_warning']):
            return 'warning'
        
        return 'healthy'
    
    def _evaluate_alerts(self, health: SystemHealth) -> List[Dict[str, Any]]:
        """アラート評価"""
        alerts = []
        thresholds = self.alert_thresholds
        
        # CPU使用率アラート
        if health.cpu_percent > thresholds['cpu_critical']:
            alerts.append({
                'type': 'cpu_critical',
                'severity': 'critical',
                'message': f'CPU usage critical: {health.cpu_percent:.1f}%',
                'value': health.cpu_percent,
                'threshold': thresholds['cpu_critical']
            })
        elif health.cpu_percent > thresholds['cpu_warning']:
            alerts.append({
                'type': 'cpu_warning',
                'severity': 'warning',
                'message': f'CPU usage high: {health.cpu_percent:.1f}%',
                'value': health.cpu_percent,
                'threshold': thresholds['cpu_warning']
            })
        
        # メモリ使用率アラート
        if health.memory_percent > thresholds['memory_critical']:
            alerts.append({
                'type': 'memory_critical',
                'severity': 'critical',
                'message': f'Memory usage critical: {health.memory_percent:.1f}%',
                'value': health.memory_percent,
                'threshold': thresholds['memory_critical']
            })
        elif health.memory_percent > thresholds['memory_warning']:
            alerts.append({
                'type': 'memory_warning',
                'severity': 'warning',
                'message': f'Memory usage high: {health.memory_percent:.1f}%',
                'value': health.memory_percent,
                'threshold': thresholds['memory_warning']
            })
        
        return alerts
    
    async def _get_db_connections(self) -> int:
        """データベース接続数取得"""
        try:
            # PostgreSQL接続数取得の実装
            # 実際の実装では psycopg2 や asyncpg を使用
            return 10  # プレースホルダー
        except Exception as e:
            logger.error(f"Error getting DB connections: {e}")
            return 0
    
    async def _collect_service_metrics(self, service_name: str) -> Optional[ServiceMetrics]:
        """サービスメトリクス収集"""
        try:
            # 実際の実装では各サービスのメトリクスエンドポイントを呼び出し
            return ServiceMetrics(
                service_name=service_name,
                timestamp=datetime.now(),
                status='healthy',
                request_rate=10.5,
                error_rate=0.1,
                avg_response_time=150.0,
                memory_usage_mb=128.5,
                cpu_usage_percent=15.2
            )
        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")
            return None
    
    async def _get_model_accuracy(self) -> Optional[float]:
        """MLモデル精度取得"""
        try:
            # 実際の実装では MLモデルの評価結果を取得
            return 0.87  # プレースホルダー
        except Exception:
            return None
    
    async def _get_data_quality_score(self) -> Optional[float]:
        """データ品質スコア取得"""
        try:
            # 実際の実装では データ品質評価システムから取得
            return 0.92  # プレースホルダー
        except Exception:
            return None
    
    async def _get_user_activity_rate(self) -> Optional[float]:
        """ユーザーアクティビティレート取得"""
        try:
            # 実際の実装では ユーザーアクティビティデータから取得
            return 156.7  # プレースホルダー
        except Exception:
            return None
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """アラート送信"""
        try:
            logger.warning(f"ALERT: {alert['message']}")
            # 実際の実装では Slack, email, PagerDuty などに送信
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def get_system_health(self) -> Optional[SystemHealth]:
        """現在のシステムヘルス取得"""
        return self.health_history[-1] if self.health_history else None
    
    def get_service_metrics(self) -> Dict[str, ServiceMetrics]:
        """サービスメトリクス取得"""
        return self.service_metrics.copy()
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """ヘルス履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.health_history if h.timestamp > cutoff_time]
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_enabled = False
        logger.info("System monitoring stopped")

# ファクトリー関数
def create_system_monitor(config: Optional[Dict[str, Any]] = None) -> SystemMonitor:
    """システム監視の作成"""
    return SystemMonitor(config)

# CLI エントリーポイント
async def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adult Matching System Monitor")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--port", type=int, default=8000, help="Prometheus metrics port")
    
    args = parser.parse_args()
    
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config['prometheus_port'] = args.port
    
    monitor = create_system_monitor(config)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    asyncio.run(main())
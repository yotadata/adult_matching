"""
Adult Matching Backend Monitoring Package

リファクタリングされたバックエンドシステムの包括的監視
- システム監視
- 統合監視
- アラート管理
- ログ統合
"""

from .system_monitor import SystemMonitor, SystemHealth, ServiceMetrics, create_system_monitor
from .integration_monitor import IntegrationMonitor, IntegrationHealth, ComponentStatus, create_integration_monitor

__version__ = "1.0.0"
__author__ = "Adult Matching Monitoring Team"

__all__ = [
    "SystemMonitor",
    "SystemHealth", 
    "ServiceMetrics",
    "IntegrationMonitor",
    "IntegrationHealth",
    "ComponentStatus",
    "create_system_monitor",
    "create_integration_monitor",
    "create_unified_monitor"
]

def create_unified_monitor(config=None):
    """
    統合監視システムの作成
    
    Args:
        config: 監視設定辞書
        
    Returns:
        tuple: (SystemMonitor, IntegrationMonitor)
    """
    system_monitor = create_system_monitor(config)
    integration_monitor = create_integration_monitor(config)
    
    return system_monitor, integration_monitor
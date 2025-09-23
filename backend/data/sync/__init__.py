"""
Data Sync Module

外部API統合データ同期システム
"""

from typing import Dict, Any, Optional

__all__ = ['create_sync_manager', 'get_available_sync_types']

def get_available_sync_types() -> Dict[str, str]:
    """利用可能な同期タイプ一覧"""
    return {
        "dmm": "DMM/FANZA API synchronization"
    }

def create_sync_manager(sync_type: str = "dmm", config: Optional[Dict[str, Any]] = None):
    """
    同期管理システムの作成
    
    Args:
        sync_type: 同期タイプ ("dmm")
        config: 同期設定
        
    Returns:
        同期管理インスタンス
    """
    if sync_type == "dmm":
        try:
            from .dmm.dmm_sync_manager import DMMSyncManager, DMMSyncConfig
            
            if config:
                sync_config = DMMSyncConfig(**config)
            else:
                sync_config = DMMSyncConfig()
                
            return DMMSyncManager(sync_config)
        except ImportError as e:
            raise ImportError(f"DMM sync manager not available: {e}")
    else:
        raise ValueError(f"Unknown sync type: {sync_type}. Available types: {list(get_available_sync_types().keys())}")
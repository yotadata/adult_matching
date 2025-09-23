"""
ML Package Logger Utilities

統一ログシステム
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from backend.ml import PACKAGE_ROOT


def get_ml_logger(name: str, 
                  level: int = logging.INFO,
                  log_file: Optional[str] = None) -> logging.Logger:
    """MLパッケージ用ロガーの取得"""
    
    logger = logging.getLogger(f"backend.ml.{name}")
    
    if logger.handlers:  # Already configured
        return logger
    
    logger.setLevel(level)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（オプション）
    if log_file:
        log_path = Path(log_file)
    else:
        log_dir = PACKAGE_ROOT / "training" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "ml_package.log"
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
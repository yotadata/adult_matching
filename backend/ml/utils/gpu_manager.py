"""
GPU Management Utilities

GPU設定と管理ユーティリティ
"""

import logging
import tensorflow as tf
from typing import List, Optional


logger = logging.getLogger(__name__)


class GPUManager:
    """GPU管理クラス"""
    
    @staticmethod
    def list_gpus() -> List[str]:
        """利用可能なGPUのリスト"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            return [gpu.name for gpu in gpus]
        except Exception as e:
            logger.warning(f"Failed to list GPUs: {e}")
            return []
    
    @staticmethod
    def setup_gpu(memory_growth: bool = True, memory_limit: Optional[int] = None) -> bool:
        """GPU設定のセットアップ"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if gpus:
                for gpu in gpus:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    if memory_limit:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )]
                        )
                
                logger.info(f"GPU setup completed: {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name}")
                return True
            else:
                logger.info("No GPU detected, using CPU")
                return False
                
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
            return False
    
    @staticmethod
    def get_gpu_info() -> dict:
        """GPU情報の取得"""
        info = {
            'available_gpus': [],
            'gpu_count': 0,
            'memory_info': [],
            'compute_capability': []
        }
        
        try:
            physical_gpus = tf.config.experimental.list_physical_devices('GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            
            info['gpu_count'] = len(physical_gpus)
            info['available_gpus'] = [gpu.name for gpu in physical_gpus]
            
            # メモリ情報取得（TensorFlow 2.x）
            for i, gpu in enumerate(physical_gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    info['memory_info'].append({
                        'gpu_id': i,
                        'name': gpu.name,
                        'details': details
                    })
                except:
                    info['memory_info'].append({
                        'gpu_id': i,
                        'name': gpu.name,
                        'details': 'Unable to get details'
                    })
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
        
        return info
    
    @staticmethod
    def set_mixed_precision(enabled: bool = True):
        """混合精度設定"""
        if enabled:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled")
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")
        else:
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision disabled")
    
    @staticmethod
    def clear_session():
        """TensorFlowセッションのクリア"""
        tf.keras.backend.clear_session()
        logger.info("TensorFlow session cleared")
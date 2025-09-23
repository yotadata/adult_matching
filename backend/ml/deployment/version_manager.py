"""
Model Version Manager

モデルバージョン管理システム
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
# import semantic_version


class ModelVersionManager:
    """モデルバージョン管理クラス"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / "versions"
        self.metadata_file = self.base_path / "version_metadata.json"
        
        # ディレクトリ初期化
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータファイル初期化
        if not self.metadata_file.exists():
            self._initialize_metadata()
    
    def _initialize_metadata(self):
        """メタデータファイル初期化"""
        initial_metadata = {
            "versions": {},
            "current_version": None,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(initial_metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """メタデータ読み込み"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """メタデータ保存"""
        metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_version(
        self,
        model_trainer,
        version_tag: str,
        description: str = "",
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """新バージョン登録"""
        try:
            # バージョン形式検証（簡易版）
            if not version_tag.replace('v', '').replace('.', '').replace('-', '').replace('_', '').isalnum():
                return {
                    'success': False,
                    'error': f"Invalid version format: {version_tag}. Use alphanumeric versioning (e.g., v1.0.0)"
                }
            
            # バージョンディレクトリ作成
            version_dir = self.versions_dir / version_tag
            version_dir.mkdir(exist_ok=True)
            
            # モデル保存
            model_path = version_dir / "model"
            model_path.mkdir(exist_ok=True)
            
            # 実際のモデル保存（簡易版）
            model_info_path = model_path / "model_info.json"
            model_info = {
                "version": version_tag,
                "model_type": "TwoTowerModel",
                "architecture": str(model_trainer.__class__.__name__),
                "saved_at": datetime.now().isoformat()
            }
            
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # バージョンメタデータ作成
            version_metadata = {
                "version": version_tag,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "model_path": str(model_path),
                "metrics": metrics or {},
                "status": "active"
            }
            
            version_metadata_path = version_dir / "metadata.json"
            with open(version_metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # グローバルメタデータ更新
            global_metadata = self._load_metadata()
            global_metadata["versions"][version_tag] = version_metadata
            
            # 最初のバージョンの場合、現在バージョンに設定
            if global_metadata["current_version"] is None:
                global_metadata["current_version"] = version_tag
            
            self._save_metadata(global_metadata)
            
            return {
                'success': True,
                'version': version_tag,
                'path': str(version_dir)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """バージョン一覧取得"""
        metadata = self._load_metadata()
        versions = []
        
        for version_tag, version_data in metadata["versions"].items():
            versions.append({
                "version": version_tag,
                "description": version_data.get("description", ""),
                "created_at": version_data.get("created_at", ""),
                "metrics": version_data.get("metrics", {}),
                "status": version_data.get("status", "unknown")
            })
        
        # バージョン順でソート（文字列ソート）
        versions.sort(key=lambda x: x["version"], reverse=True)
        
        return versions
    
    def get_version(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """特定バージョン取得"""
        metadata = self._load_metadata()
        return metadata["versions"].get(version_tag)
    
    def get_latest_version(self) -> Optional[Dict[str, Any]]:
        """最新バージョン取得"""
        versions = self.list_versions()
        return versions[0] if versions else None
    
    def get_current_version(self) -> Optional[Dict[str, Any]]:
        """現在バージョン取得"""
        metadata = self._load_metadata()
        current_version_tag = metadata.get("current_version")
        
        if current_version_tag:
            return self.get_version(current_version_tag)
        return None
    
    def set_current_version(self, version_tag: str) -> Dict[str, Any]:
        """現在バージョン設定"""
        try:
            metadata = self._load_metadata()
            
            if version_tag not in metadata["versions"]:
                return {
                    'success': False,
                    'error': f"Version {version_tag} not found"
                }
            
            metadata["current_version"] = version_tag
            self._save_metadata(metadata)
            
            return {
                'success': True,
                'current_version': version_tag
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def rollback_to_version(self, version_tag: str) -> Dict[str, Any]:
        """指定バージョンへのロールバック"""
        return self.set_current_version(version_tag)
    
    def delete_version(self, version_tag: str) -> Dict[str, Any]:
        """バージョン削除"""
        try:
            metadata = self._load_metadata()
            
            if version_tag not in metadata["versions"]:
                return {
                    'success': False,
                    'error': f"Version {version_tag} not found"
                }
            
            # 現在バージョンの場合は削除不可
            if metadata.get("current_version") == version_tag:
                return {
                    'success': False,
                    'error': f"Cannot delete current version {version_tag}"
                }
            
            # ディレクトリ削除
            version_dir = self.versions_dir / version_tag
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # メタデータから削除
            del metadata["versions"][version_tag]
            self._save_metadata(metadata)
            
            return {
                'success': True,
                'deleted_version': version_tag
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """バージョン比較"""
        try:
            metadata = self._load_metadata()
            
            version_a_data = metadata["versions"].get(version_a)
            version_b_data = metadata["versions"].get(version_b)
            
            if not version_a_data or not version_b_data:
                return {
                    'success': False,
                    'error': "One or both versions not found"
                }
            
            # 文字列比較
            if version_a > version_b:
                newer, older = version_a, version_b
            elif version_b > version_a:
                newer, older = version_b, version_a
            else:
                newer, older = None, None
            
            # メトリクス差分
            metrics_a = version_a_data.get("metrics", {})
            metrics_b = version_b_data.get("metrics", {})
            
            metrics_diff = {}
            all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
            
            for metric in all_metrics:
                val_a = metrics_a.get(metric, 0)
                val_b = metrics_b.get(metric, 0)
                
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    metrics_diff[metric] = {
                        f'{version_a}': val_a,
                        f'{version_b}': val_b,
                        'difference': val_a - val_b
                    }
                else:
                    metrics_diff[metric] = {
                        f'{version_a}': val_a,
                        f'{version_b}': val_b,
                        'difference': 'N/A (non-numeric)'
                    }
            
            return {
                'success': True,
                'newer': newer,
                'older': older,
                'same_version': newer is None and older is None,
                'metrics_diff': metrics_diff,
                'version_a_data': version_a_data,
                'version_b_data': version_b_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
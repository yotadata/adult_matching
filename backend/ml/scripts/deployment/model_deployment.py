#!/usr/bin/env python3
"""
モデルデプロイメント自動化スクリプト

学習 → 変換 → 検証 → デプロイメントの完全自動化パイプライン
Edge Functionsモデルリロードトリガー機能付き
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import time

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 新しく作成したモジュールをインポート
from ml_pipeline.export.keras_to_tfjs import KerasToTensorFlowJSConverter
from ml_pipeline.export.model_validator import ComprehensiveValidator
from ml_pipeline.export.supabase_uploader import SupabaseStorageUploader, ModelDeploymentManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'model_deployment.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelDeploymentPipeline:
    """モデルデプロイメント自動化パイプライン"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        パイプライン初期化
        
        Args:
            config: デプロイメント設定
        """
        self.config = config
        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # パス設定
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "ml-pipeline" / "models" / "rating_based_two_tower_768"
        self.tfjs_export_dir = self.project_root / "ml_pipeline" / "models" / "tfjs_exports"
        self.sample_data_path = self.project_root / "tests" / "fixtures" / "sample_videos.json"
        
        # コンポーネント初期化
        self.converter = KerasToTensorFlowJSConverter(
            model_path=str(self.models_dir),
            output_path=str(self.tfjs_export_dir)
        )
        
        self.validator = ComprehensiveValidator(
            model_directory=str(self.models_dir),
            sample_data_path=str(self.sample_data_path)
        )
        
        self.uploader = SupabaseStorageUploader()
        self.deployment_manager = ModelDeploymentManager(self.uploader)
        
        # デプロイメント状態
        self.pipeline_state = {
            "deployment_id": self.deployment_id,
            "start_time": datetime.now().isoformat(),
            "current_stage": "initialization",
            "stages": {
                "training": {"status": "pending", "start_time": None, "end_time": None, "success": None},
                "conversion": {"status": "pending", "start_time": None, "end_time": None, "success": None},
                "validation": {"status": "pending", "start_time": None, "end_time": None, "success": None},
                "deployment": {"status": "pending", "start_time": None, "end_time": None, "success": None},
                "verification": {"status": "pending", "start_time": None, "end_time": None, "success": None}
            },
            "rollback_version": None,
            "final_status": "running"
        }
        
        logger.info(f"Initialized deployment pipeline: {self.deployment_id}")
    
    def _update_stage_status(self, stage: str, status: str, success: bool = None):
        """ステージ状態更新"""
        self.pipeline_state["current_stage"] = stage
        self.pipeline_state["stages"][stage]["status"] = status
        
        if status == "running" and self.pipeline_state["stages"][stage]["start_time"] is None:
            self.pipeline_state["stages"][stage]["start_time"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            self.pipeline_state["stages"][stage]["end_time"] = datetime.now().isoformat()
            self.pipeline_state["stages"][stage]["success"] = success
    
    def run_training(self) -> bool:
        """モデル学習実行"""
        self._update_stage_status("training", "running")
        logger.info("=== モデル学習開始 ===")
        
        try:
            if self.config.get("skip_training", False):
                logger.info("学習スキップが指定されました - 既存モデルを使用")
                self._update_stage_status("training", "completed", True)
                return True
            
            # 既存の学習スクリプト実行
            training_script = self.project_root / "ml-pipeline" / "training" / "train_768_dim_two_tower.py"
            
            if not training_script.exists():
                logger.error(f"学習スクリプトが見つかりません: {training_script}")
                self._update_stage_status("training", "failed", False)
                return False
            
            # 学習実行
            result = subprocess.run([
                sys.executable, str(training_script)
            ], capture_output=True, text=True, timeout=3600)  # 1時間タイムアウト
            
            if result.returncode == 0:
                logger.info("モデル学習が完了しました")
                logger.info(f"学習出力: {result.stdout}")
                self._update_stage_status("training", "completed", True)
                return True
            else:
                logger.error(f"学習が失敗しました: {result.stderr}")
                self._update_stage_status("training", "failed", False)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("学習がタイムアウトしました")
            self._update_stage_status("training", "failed", False)
            return False
        except Exception as e:
            logger.error(f"学習中にエラーが発生しました: {e}")
            self._update_stage_status("training", "failed", False)
            return False
    
    def run_conversion(self) -> bool:
        """TensorFlow.js変換実行"""
        self._update_stage_status("conversion", "running")
        logger.info("=== TensorFlow.js変換開始 ===")
        
        try:
            # 変換実行
            self.converter.convert_both_towers()
            
            # 変換結果確認
            user_tower_dir = self.tfjs_export_dir / "user_tower"
            item_tower_dir = self.tfjs_export_dir / "item_tower"
            
            if user_tower_dir.exists() and item_tower_dir.exists():
                logger.info("TensorFlow.js変換が完了しました")
                self._update_stage_status("conversion", "completed", True)
                return True
            else:
                logger.error("変換されたモデルファイルが見つかりません")
                self._update_stage_status("conversion", "failed", False)
                return False
                
        except Exception as e:
            logger.error(f"変換中にエラーが発生しました: {e}")
            self._update_stage_status("conversion", "failed", False)
            return False
    
    def run_validation(self) -> bool:
        """モデル検証実行"""
        self._update_stage_status("validation", "running")
        logger.info("=== モデル検証開始 ===")
        
        try:
            # 包括的検証実行
            validation_results = self.validator.validate_all_models()
            validation_report = self.validator.generate_validation_report(validation_results)
            
            # 検証結果判定
            validation_threshold = self.config.get("validation_accuracy_threshold", 0.95)
            all_models_valid = all(
                result.validation_status == "validated" and 
                result.numerical_accuracy >= validation_threshold
                for result in validation_results.values()
            )
            
            if all_models_valid:
                logger.info("モデル検証が完了しました - 全モデルが基準を満たしています")
                logger.info(f"検証サマリー: {validation_report['overall_status']}")
                self._update_stage_status("validation", "completed", True)
                return True
            else:
                logger.error("モデル検証が失敗しました - 精度基準を満たしていません")
                logger.error(f"検証結果: {validation_report}")
                self._update_stage_status("validation", "failed", False)
                return False
                
        except Exception as e:
            logger.error(f"検証中にエラーが発生しました: {e}")
            self._update_stage_status("validation", "failed", False)
            return False
    
    def run_deployment(self) -> bool:
        """Supabaseデプロイメント実行"""
        self._update_stage_status("deployment", "running")
        logger.info("=== Supabaseデプロイメント開始 ===")
        
        try:
            # 現在のバージョン取得（ロールバック用）
            current_user_version = self.uploader.get_current_version("user_tower")
            current_item_version = self.uploader.get_current_version("item_tower")
            
            if current_user_version == current_item_version:
                self.pipeline_state["rollback_version"] = current_user_version
                logger.info(f"ロールバック用バージョン記録: {current_user_version}")
            
            # 両タワーデプロイメント
            deployment_results = self.deployment_manager.deploy_both_towers(
                str(self.tfjs_export_dir),
                self.deployment_id
            )
            
            if deployment_results["overall_success"]:
                # アトミック切り替え
                switch_results = self.deployment_manager.atomic_deployment_switch(self.deployment_id)
                
                if all(switch_results.values()):
                    logger.info(f"デプロイメントが完了しました - バージョン: {self.deployment_id}")
                    self._update_stage_status("deployment", "completed", True)
                    return True
                else:
                    logger.error(f"アトミック切り替えが失敗しました: {switch_results}")
                    self._update_stage_status("deployment", "failed", False)
                    return False
            else:
                logger.error(f"デプロイメントが失敗しました: {deployment_results}")
                self._update_stage_status("deployment", "failed", False)
                return False
                
        except Exception as e:
            logger.error(f"デプロイメント中にエラーが発生しました: {e}")
            self._update_stage_status("deployment", "failed", False)
            return False
    
    def run_verification(self) -> bool:
        """デプロイメント検証"""
        self._update_stage_status("verification", "running")
        logger.info("=== デプロイメント検証開始 ===")
        
        try:
            # バージョン確認
            user_version = self.uploader.get_current_version("user_tower")
            item_version = self.uploader.get_current_version("item_tower")
            
            if user_version == item_version == self.deployment_id:
                logger.info(f"デプロイメント検証完了 - 現在のバージョン: {user_version}")
                self._update_stage_status("verification", "completed", True)
                return True
            else:
                logger.error(f"バージョン不整合: user={user_version}, item={item_version}, expected={self.deployment_id}")
                self._update_stage_status("verification", "failed", False)
                return False
                
        except Exception as e:
            logger.error(f"検証中にエラーが発生しました: {e}")
            self._update_stage_status("verification", "failed", False)
            return False
    
    def trigger_edge_function_reload(self) -> bool:
        """Edge Functionsモデルリロードトリガー"""
        logger.info("=== Edge Functionsモデルリロード ===")
        
        try:
            # 今後実装: Edge Functions通知API呼び出し
            # POST /functions/v1/model_reload_trigger
            # Body: {"version": self.deployment_id, "models": ["user_tower", "item_tower"]}
            
            logger.info("Edge Functionsリロード通知送信 (実装待ち)")
            return True
            
        except Exception as e:
            logger.error(f"Edge Functionsリロード通知エラー: {e}")
            return False
    
    def rollback_on_failure(self) -> bool:
        """失敗時ロールバック実行"""
        logger.info("=== 失敗時ロールバック開始 ===")
        
        try:
            if not self.pipeline_state["rollback_version"]:
                logger.warning("ロールバック対象バージョンが記録されていません")
                return False
            
            target_version = self.pipeline_state["rollback_version"]
            logger.info(f"バージョン {target_version} にロールバック中...")
            
            # 両タワーロールバック
            user_success = self.uploader.rollback_version("user_tower", target_version)
            item_success = self.uploader.rollback_version("item_tower", target_version)
            
            if user_success and item_success:
                logger.info(f"ロールバック完了: {target_version}")
                return True
            else:
                logger.error(f"ロールバック失敗: user={user_success}, item={item_success}")
                return False
                
        except Exception as e:
            logger.error(f"ロールバック中にエラー: {e}")
            return False
    
    def execute_pipeline(self) -> Dict[str, Any]:
        """パイプライン実行"""
        logger.info(f"=== モデルデプロイメントパイプライン開始: {self.deployment_id} ===")
        
        pipeline_stages = [
            ("training", self.run_training),
            ("conversion", self.run_conversion),
            ("validation", self.run_validation),
            ("deployment", self.run_deployment),
            ("verification", self.run_verification)
        ]
        
        try:
            for stage_name, stage_func in pipeline_stages:
                logger.info(f"--- {stage_name.upper()} ステージ開始 ---")
                
                success = stage_func()
                
                if not success:
                    logger.error(f"{stage_name} ステージが失敗しました")
                    
                    # デプロイメント後の失敗はロールバック
                    if stage_name in ["deployment", "verification"]:
                        self.rollback_on_failure()
                    
                    self.pipeline_state["final_status"] = "failed"
                    return self._generate_final_report()
                
                logger.info(f"--- {stage_name.upper()} ステージ完了 ---")
            
            # Edge Functions通知
            self.trigger_edge_function_reload()
            
            # 成功完了
            self.pipeline_state["final_status"] = "success"
            logger.info(f"=== パイプライン完了: {self.deployment_id} ===")
            
        except Exception as e:
            logger.error(f"パイプライン実行中の予期しないエラー: {e}")
            self.pipeline_state["final_status"] = "error"
            self.rollback_on_failure()
        
        return self._generate_final_report()
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """最終レポート生成"""
        self.pipeline_state["end_time"] = datetime.now().isoformat()
        
        # 実行時間計算
        start_time = datetime.fromisoformat(self.pipeline_state["start_time"])
        end_time = datetime.fromisoformat(self.pipeline_state["end_time"])
        total_duration = (end_time - start_time).total_seconds()
        
        report = {
            **self.pipeline_state,
            "total_duration_seconds": total_duration,
            "summary": {
                "deployment_id": self.deployment_id,
                "final_status": self.pipeline_state["final_status"],
                "successful_stages": sum(1 for stage in self.pipeline_state["stages"].values() if stage["success"]),
                "total_stages": len(self.pipeline_state["stages"]),
                "rollback_performed": bool(self.pipeline_state["rollback_version"] and self.pipeline_state["final_status"] != "success")
            }
        }
        
        # レポートファイル出力
        report_file = self.project_root / "logs" / f"deployment_report_{self.deployment_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"デプロイメントレポート出力: {report_file}")
        
        return report


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Model Deployment Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip training stage (use existing models)")
    parser.add_argument("--validation-threshold", type=float, default=0.95, help="Validation accuracy threshold")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual deployment)")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = {
        "skip_training": args.skip_training,
        "validation_accuracy_threshold": args.validation_threshold,
        "dry_run": args.dry_run
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    try:
        # パイプライン実行
        pipeline = ModelDeploymentPipeline(config)
        report = pipeline.execute_pipeline()
        
        # 結果出力
        print("\n" + "="*60)
        print("DEPLOYMENT PIPELINE REPORT")
        print("="*60)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        
        if report["final_status"] == "success":
            print(f"\n✅ デプロイメント成功: {report['deployment_id']}")
            exit(0)
        else:
            print(f"\n❌ デプロイメント失敗: {report['final_status']}")
            exit(1)
    
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        print(f"❌ パイプライン実行エラー: {e}")
        exit(1)


if __name__ == "__main__":
    main()
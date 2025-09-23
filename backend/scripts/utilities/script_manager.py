"""
Script Manager

統合スクリプト管理システム - 全スクリプトの実行・監視・ログ管理
"""

import asyncio
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class ScriptCategory(Enum):
    """スクリプトカテゴリ"""
    ML_TRAINING = "ml_training"
    DATA_INGESTION = "data_ingestion"
    DATA_ANALYSIS = "data_analysis"
    DATA_PROCESSING = "data_processing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    TESTING = "testing"


class ScriptStatus(Enum):
    """スクリプト実行状態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScriptDefinition:
    """スクリプト定義"""
    name: str
    path: Path
    category: ScriptCategory
    description: str
    language: str  # python, node, bash
    dependencies: List[str] = field(default_factory=list)
    arguments: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 3600
    retry_count: int = 0


@dataclass
class ScriptExecution:
    """スクリプト実行結果"""
    script_name: str
    execution_id: str
    status: ScriptStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ScriptManager:
    """統合スクリプト管理システム"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.logger = logging.getLogger(__name__)
        self.scripts: Dict[str, ScriptDefinition] = {}
        self.executions: Dict[str, ScriptExecution] = {}
        self._load_script_registry()
    
    def register_script(self, script_def: ScriptDefinition):
        """スクリプト登録"""
        self.scripts[script_def.name] = script_def
        self.logger.info(f"スクリプト登録: {script_def.name} ({script_def.category.value})")
    
    def _load_script_registry(self):
        """スクリプトレジストリ読み込み"""
        # ML Training Scripts
        self.register_script(ScriptDefinition(
            name="train_production_two_tower",
            path=self.base_path.parent / "ml" / "training" / "scripts" / "train_production_two_tower.py",
            category=ScriptCategory.ML_TRAINING,
            description="本番Two-Towerモデルトレーニング",
            language="python",
            timeout_seconds=7200  # 2 hours
        ))
        
        self.register_script(ScriptDefinition(
            name="train_768_dim_two_tower", 
            path=self.base_path.parent / "ml" / "training" / "scripts" / "train_768_dim_two_tower.py",
            category=ScriptCategory.ML_TRAINING,
            description="768次元Two-Towerモデルトレーニング",
            language="python",
            timeout_seconds=3600  # 1 hour
        ))
        
        self.register_script(ScriptDefinition(
            name="verify_768_model",
            path=self.base_path.parent / "ml" / "training" / "scripts" / "verify_768_model.py",
            category=ScriptCategory.TESTING,
            description="768次元モデル検証",
            language="python",
            timeout_seconds=600  # 10 minutes
        ))
        
        self.register_script(ScriptDefinition(
            name="standardize_models_768",
            path=self.base_path.parent / "ml" / "scripts" / "standardize_models_768.py",
            category=ScriptCategory.DEPLOYMENT,
            description="768次元モデル標準化とデプロイメント",
            language="python",
            timeout_seconds=1200  # 20 minutes
        ))
        
        # Data Ingestion Scripts
        self.register_script(ScriptDefinition(
            name="dmm_api_sync",
            path=self.base_path.parent / "data" / "ingestion" / "scripts" / "dmm_api_sync.js",
            category=ScriptCategory.DATA_INGESTION,
            description="DMM API データ同期",
            language="node",
            timeout_seconds=3600  # 1 hour
        ))
        
        # Data Analysis Scripts
        self.register_script(ScriptDefinition(
            name="analyze_dmm_data",
            path=self.base_path.parent / "data" / "analysis" / "scripts" / "analyze_dmm_data.js",
            category=ScriptCategory.DATA_ANALYSIS,
            description="DMM データ品質分析",
            language="node",
            timeout_seconds=1800  # 30 minutes
        ))
        
        self.register_script(ScriptDefinition(
            name="analyze_review_dates",
            path=self.base_path.parent / "data" / "analysis" / "scripts" / "analyze_review_dates.js",
            category=ScriptCategory.DATA_ANALYSIS,
            description="レビュー日付分析",
            language="node",
            timeout_seconds=600  # 10 minutes
        ))
    
    async def execute_script(self, script_name: str, 
                           arguments: Optional[Dict[str, Any]] = None,
                           environment: Optional[Dict[str, str]] = None) -> ScriptExecution:
        """スクリプト実行"""
        if script_name not in self.scripts:
            raise ValueError(f"スクリプト '{script_name}' が見つかりません")
        
        script_def = self.scripts[script_name]
        execution_id = f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution = ScriptExecution(
            script_name=script_name,
            execution_id=execution_id,
            status=ScriptStatus.PENDING,
            start_time=datetime.now()
        )
        
        self.executions[execution_id] = execution
        
        try:
            execution.status = ScriptStatus.RUNNING
            self.logger.info(f"スクリプト実行開始: {script_name}")
            
            # コマンド構築
            if script_def.language == "python":
                cmd = ["python", str(script_def.path)]
            elif script_def.language == "node":
                cmd = ["node", str(script_def.path)]
            elif script_def.language == "bash":
                cmd = ["bash", str(script_def.path)]
            else:
                raise ValueError(f"未対応言語: {script_def.language}")
            
            # 引数追加
            if arguments:
                for key, value in arguments.items():
                    cmd.extend([f"--{key}", str(value)])
            
            # 環境変数設定
            env = script_def.environment.copy()
            if environment:
                env.update(environment)
            
            # プロセス実行
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=script_def.path.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=script_def.timeout_seconds
                )
                
                execution.exit_code = process.returncode
                execution.stdout = stdout.decode('utf-8') if stdout else ""
                execution.stderr = stderr.decode('utf-8') if stderr else ""
                
                if process.returncode == 0:
                    execution.status = ScriptStatus.COMPLETED
                    self.logger.info(f"スクリプト実行完了: {script_name}")
                else:
                    execution.status = ScriptStatus.FAILED
                    execution.error_message = f"Exit code: {process.returncode}"
                    self.logger.error(f"スクリプト実行失敗: {script_name}, exit code: {process.returncode}")
                
            except asyncio.TimeoutError:
                process.kill()
                execution.status = ScriptStatus.FAILED
                execution.error_message = f"タイムアウト ({script_def.timeout_seconds}秒)"
                self.logger.error(f"スクリプトタイムアウト: {script_name}")
                
        except Exception as e:
            execution.status = ScriptStatus.FAILED
            execution.error_message = str(e)
            self.logger.error(f"スクリプト実行エラー: {script_name} - {e}")
        
        finally:
            execution.end_time = datetime.now()
        
        return execution
    
    def get_script_status(self, execution_id: str) -> Optional[ScriptExecution]:
        """スクリプト実行状況取得"""
        return self.executions.get(execution_id)
    
    def list_scripts(self, category: Optional[ScriptCategory] = None) -> List[ScriptDefinition]:
        """スクリプト一覧取得"""
        scripts = list(self.scripts.values())
        if category:
            scripts = [s for s in scripts if s.category == category]
        return scripts
    
    def get_execution_history(self, script_name: Optional[str] = None, 
                            limit: int = 50) -> List[ScriptExecution]:
        """実行履歴取得"""
        executions = list(self.executions.values())
        if script_name:
            executions = [e for e in executions if e.script_name == script_name]
        
        executions.sort(key=lambda x: x.start_time, reverse=True)
        return executions[:limit]
    
    async def run_ml_training_pipeline(self) -> List[ScriptExecution]:
        """ML トレーニングパイプライン実行"""
        pipeline_scripts = [
            "verify_768_model",
            "train_768_dim_two_tower",
            "train_production_two_tower",
            "standardize_models_768"
        ]
        
        results = []
        for script_name in pipeline_scripts:
            self.logger.info(f"パイプライン実行: {script_name}")
            result = await self.execute_script(script_name)
            results.append(result)
            
            if result.status != ScriptStatus.COMPLETED:
                self.logger.error(f"パイプライン中断: {script_name} 失敗")
                break
        
        return results
    
    async def run_data_sync_pipeline(self) -> List[ScriptExecution]:
        """データ同期パイプライン実行"""
        pipeline_scripts = [
            "dmm_api_sync",
            "analyze_dmm_data"
        ]
        
        results = []
        for script_name in pipeline_scripts:
            self.logger.info(f"データ同期パイプライン実行: {script_name}")
            result = await self.execute_script(script_name)
            results.append(result)
            
            if result.status != ScriptStatus.COMPLETED:
                self.logger.error(f"データ同期パイプライン中断: {script_name} 失敗")
                break
        
        return results
    
    def export_execution_log(self, output_path: Path):
        """実行ログエクスポート"""
        log_data = {
            "export_time": datetime.now().isoformat(),
            "total_executions": len(self.executions),
            "executions": [
                {
                    "execution_id": e.execution_id,
                    "script_name": e.script_name,
                    "status": e.status.value,
                    "start_time": e.start_time.isoformat(),
                    "end_time": e.end_time.isoformat() if e.end_time else None,
                    "duration_seconds": e.duration_seconds,
                    "exit_code": e.exit_code,
                    "error_message": e.error_message
                }
                for e in self.executions.values()
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"実行ログエクスポート完了: {output_path}")


# CLI インターフェース
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Script Manager")
        parser.add_argument("--list", action="store_true", help="スクリプト一覧表示")
        parser.add_argument("--run", type=str, help="スクリプト実行")
        parser.add_argument("--category", type=str, help="カテゴリフィルタ")
        parser.add_argument("--history", action="store_true", help="実行履歴表示")
        parser.add_argument("--ml-pipeline", action="store_true", help="MLパイプライン実行")
        parser.add_argument("--data-pipeline", action="store_true", help="データパイプライン実行")
        
        args = parser.parse_args()
        
        manager = ScriptManager()
        
        if args.list:
            category = ScriptCategory(args.category) if args.category else None
            scripts = manager.list_scripts(category)
            print("登録済みスクリプト:")
            for script in scripts:
                print(f"  {script.name} ({script.category.value}) - {script.description}")
        
        elif args.run:
            result = await manager.execute_script(args.run)
            print(f"実行結果: {result.status.value}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
        
        elif args.ml_pipeline:
            results = await manager.run_ml_training_pipeline()
            print(f"MLパイプライン実行完了: {len(results)}スクリプト")
            for result in results:
                print(f"  {result.script_name}: {result.status.value}")
        
        elif args.data_pipeline:
            results = await manager.run_data_sync_pipeline()
            print(f"データパイプライン実行完了: {len(results)}スクリプト")
            for result in results:
                print(f"  {result.script_name}: {result.status.value}")
        
        elif args.history:
            history = manager.get_execution_history(limit=10)
            print("実行履歴 (最新10件):")
            for execution in history:
                status = execution.status.value
                duration = f"{execution.duration_seconds:.1f}s" if execution.duration_seconds else "実行中"
                print(f"  {execution.start_time.strftime('%Y-%m-%d %H:%M')} - {execution.script_name} ({status}, {duration})")
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
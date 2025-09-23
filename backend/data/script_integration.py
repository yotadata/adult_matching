"""
Script Integration for Unified Data Management

統合データ管理システムとスクリプト管理の統合
- スクリプト実行と監視
- データパイプラインとスクリプトの協調
- 統合レポートと管理インターフェース
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

# スクリプト管理システムの遅延インポート
def get_script_migration_manager():
    """スクリプト移行管理システムの取得"""
    try:
        from ..scripts.script_migration_manager import ScriptMigrationManager
        return ScriptMigrationManager
    except ImportError:
        logger.warning("Script migration manager not available")
        return None

def get_script_runner():
    """統合スクリプト実行システムの取得"""
    try:
        from ..scripts.unified_script_runner import ScriptRunner
        return ScriptRunner
    except ImportError:
        logger.warning("Unified script runner not available")
        return None

@dataclass
class ScriptExecutionResult:
    """スクリプト実行結果"""
    script_name: str
    exit_code: int
    execution_time: float
    start_time: datetime
    end_time: datetime
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = None

class IntegratedScriptManager:
    """統合スクリプト管理システム"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.script_runner = None
        self.migration_manager = None
        self.execution_history: List[ScriptExecutionResult] = []
        
        # スクリプト管理コンポーネントの初期化
        self._initialize_script_components()
        
        logger.info(f"Integrated script manager initialized for {self.project_root}")
    
    def _initialize_script_components(self):
        """スクリプト管理コンポーネントの初期化"""
        try:
            script_runner_class = get_script_runner()
            if script_runner_class:
                self.script_runner = script_runner_class()
                logger.info("Script runner initialized successfully")
            
            migration_manager_class = get_script_migration_manager()
            if migration_manager_class:
                self.migration_manager = migration_manager_class(self.project_root)
                logger.info("Migration manager initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize script components: {e}")
    
    async def run_data_processing_pipeline(self, pipeline_name: str = "data_quality") -> ScriptExecutionResult:
        """データ処理パイプラインの実行"""
        if not self.script_runner:
            raise ValueError("Script runner not available")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data processing pipeline: {pipeline_name}")
            exit_code = self.script_runner.run_pipeline(pipeline_name)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = ScriptExecutionResult(
                script_name=f"pipeline_{pipeline_name}",
                exit_code=exit_code,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output=f"Pipeline {pipeline_name} executed successfully" if exit_code == 0 else f"Pipeline {pipeline_name} failed",
                metadata={"pipeline_type": "data_processing", "pipeline_name": pipeline_name}
            )
            
            self.execution_history.append(result)
            
            if exit_code == 0:
                logger.info(f"Pipeline {pipeline_name} completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"Pipeline {pipeline_name} failed with exit code {exit_code}")
            
            return result
        
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = ScriptExecutionResult(
                script_name=f"pipeline_{pipeline_name}",
                exit_code=1,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                metadata={"pipeline_type": "data_processing", "pipeline_name": pipeline_name}
            )
            
            self.execution_history.append(result)
            logger.error(f"Pipeline execution failed: {e}")
            return result
    
    async def run_script_by_name(self, script_name: str, args: List[str] = None) -> ScriptExecutionResult:
        """名前によるスクリプト実行"""
        if not self.script_runner:
            raise ValueError("Script runner not available")
        
        start_time = datetime.now()
        args = args or []
        
        try:
            logger.info(f"Executing script: {script_name} with args: {args}")
            exit_code = self.script_runner.run_script(script_name, *args)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = ScriptExecutionResult(
                script_name=script_name,
                exit_code=exit_code,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output=f"Script {script_name} executed successfully" if exit_code == 0 else f"Script {script_name} failed",
                metadata={"execution_type": "direct", "args": args}
            )
            
            self.execution_history.append(result)
            
            if exit_code == 0:
                logger.info(f"Script {script_name} completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"Script {script_name} failed with exit code {exit_code}")
            
            return result
        
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = ScriptExecutionResult(
                script_name=script_name,
                exit_code=1,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=str(e),
                metadata={"execution_type": "direct", "args": args}
            )
            
            self.execution_history.append(result)
            logger.error(f"Script execution failed: {e}")
            return result
    
    def get_available_scripts(self) -> Dict[str, Any]:
        """利用可能なスクリプトの取得"""
        if not self.script_runner:
            return {}
        
        # スクリプトレジストリから情報を取得
        return self.script_runner.script_registry
    
    def get_available_pipelines(self) -> List[str]:
        """利用可能なパイプラインの取得"""
        return ["ml_full", "data_sync", "full_build", "data_quality"]
    
    def search_scripts(self, query: str) -> List[Dict[str, str]]:
        """スクリプト検索"""
        if not self.script_runner:
            return []
        
        results = []
        for category_name, category in self.script_runner.script_registry.items():
            for subcategory_name, subcategory in category.items():
                for script_name, script_info in subcategory.items():
                    if (query.lower() in script_name.lower() or 
                        query.lower() in script_info.get("description", "").lower()):
                        results.append({
                            "category": category_name,
                            "subcategory": subcategory_name,
                            "name": script_name,
                            "description": script_info.get("description", ""),
                            "path": script_info.get("path", ""),
                            "type": script_info.get("type", "")
                        })
        
        return results
    
    async def perform_script_audit(self) -> Dict[str, Any]:
        """スクリプト監査の実行"""
        if not self.migration_manager:
            logger.warning("Migration manager not available for audit")
            return {}
        
        try:
            logger.info("Starting comprehensive script audit...")
            audit_results = self.migration_manager.audit_all_scripts()
            logger.info(f"Script audit completed: {audit_results.get('total_scripts', 0)} scripts analyzed")
            return audit_results
        
        except Exception as e:
            logger.error(f"Script audit failed: {e}")
            return {"error": str(e)}
    
    async def generate_migration_plan(self) -> List[Dict[str, Any]]:
        """移行計画の生成"""
        if not self.migration_manager:
            logger.warning("Migration manager not available for migration planning")
            return []
        
        try:
            logger.info("Generating script migration plans...")
            plans = self.migration_manager.generate_migration_plans()
            logger.info(f"Generated {len(plans)} migration plans")
            
            return [
                {
                    "source_path": str(plan.source_script.path),
                    "target_path": str(plan.target_location),
                    "migration_type": plan.migration_type,
                    "priority": plan.priority,
                    "estimated_effort": plan.estimated_effort,
                    "status": plan.source_script.migration_status.value,
                    "description": plan.source_script.description
                }
                for plan in plans
            ]
        
        except Exception as e:
            logger.error(f"Migration plan generation failed: {e}")
            return []
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """実行履歴の取得"""
        recent_executions = self.execution_history[-limit:] if limit > 0 else self.execution_history
        
        return [
            {
                "script_name": result.script_name,
                "exit_code": result.exit_code,
                "execution_time": result.execution_time,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "status": "success" if result.exit_code == 0 else "failed",
                "output": result.output[:200] + "..." if len(result.output) > 200 else result.output,
                "error": result.error[:200] + "..." if len(result.error) > 200 else result.error,
                "metadata": result.metadata or {}
            }
            for result in reversed(recent_executions)
        ]
    
    def get_script_statistics(self) -> Dict[str, Any]:
        """スクリプト統計の取得"""
        if not self.execution_history:
            return {"total_executions": 0, "success_rate": 0.0, "average_execution_time": 0.0}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.exit_code == 0)
        success_rate = (successful_executions / total_executions) * 100
        
        average_execution_time = sum(result.execution_time for result in self.execution_history) / total_executions
        
        # 最近の実行統計
        recent_executions = self.execution_history[-10:] if len(self.execution_history) > 10 else self.execution_history
        recent_success_rate = (sum(1 for result in recent_executions if result.exit_code == 0) / len(recent_executions)) * 100 if recent_executions else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": total_executions - successful_executions,
            "success_rate": round(success_rate, 2),
            "recent_success_rate": round(recent_success_rate, 2),
            "average_execution_time": round(average_execution_time, 2),
            "last_execution": self.execution_history[-1].start_time.isoformat() if self.execution_history else None
        }
    
    async def run_integrated_data_workflow(self, workflow_name: str = "comprehensive") -> Dict[str, Any]:
        """統合データワークフローの実行"""
        workflow_results = {
            "workflow_name": workflow_name,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "overall_status": "started"
        }
        
        try:
            if workflow_name == "comprehensive":
                # 包括的データワークフロー
                steps = [
                    ("audit", "スクリプト監査"),
                    ("data_sync", "データ同期パイプライン"),
                    ("data_quality", "データ品質チェック"),
                    ("ml_training", "ML訓練準備")
                ]
                
                for step_id, step_name in steps:
                    step_start = datetime.now()
                    logger.info(f"Executing workflow step: {step_name}")
                    
                    try:
                        if step_id == "audit":
                            result = await self.perform_script_audit()
                            step_status = "success" if result and not result.get("error") else "failed"
                        else:
                            execution_result = await self.run_data_processing_pipeline(step_id)
                            step_status = "success" if execution_result.exit_code == 0 else "failed"
                    
                    except Exception as e:
                        step_status = "failed"
                        result = {"error": str(e)}
                    
                    step_end = datetime.now()
                    step_duration = (step_end - step_start).total_seconds()
                    
                    workflow_results["steps"].append({
                        "step_id": step_id,
                        "step_name": step_name,
                        "status": step_status,
                        "duration": step_duration,
                        "start_time": step_start.isoformat(),
                        "end_time": step_end.isoformat()
                    })
                    
                    if step_status == "failed":
                        logger.error(f"Workflow step {step_name} failed, stopping workflow")
                        workflow_results["overall_status"] = "failed"
                        break
                
                if workflow_results["overall_status"] != "failed":
                    workflow_results["overall_status"] = "completed"
            
            else:
                # カスタムワークフロー（単一パイプライン実行）
                result = await self.run_data_processing_pipeline(workflow_name)
                workflow_results["steps"].append({
                    "step_id": workflow_name,
                    "step_name": f"{workflow_name} pipeline",
                    "status": "success" if result.exit_code == 0 else "failed",
                    "duration": result.execution_time,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat()
                })
                workflow_results["overall_status"] = "completed" if result.exit_code == 0 else "failed"
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow_results["overall_status"] = "failed"
            workflow_results["error"] = str(e)
        
        workflow_results["end_time"] = datetime.now().isoformat()
        total_duration = sum(step.get("duration", 0) for step in workflow_results["steps"])
        workflow_results["total_duration"] = total_duration
        
        logger.info(f"Workflow {workflow_name} completed with status: {workflow_results['overall_status']}")
        return workflow_results

# ファクトリー関数
def create_integrated_script_manager(project_root: Optional[Path] = None) -> IntegratedScriptManager:
    """統合スクリプト管理システムの作成"""
    return IntegratedScriptManager(project_root)
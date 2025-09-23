#!/usr/bin/env python3
"""
Script Migration Manager
スクリプト監査・分類・移行管理システム
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import shutil
import subprocess
import ast
import re

logger = logging.getLogger(__name__)

class ScriptLanguage(Enum):
    """スクリプト言語"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    SHELL = "shell"
    UNKNOWN = "unknown"

class ScriptCategory(Enum):
    """スクリプトカテゴリ"""
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    DATA_SYNC = "data_sync"
    DATA_ANALYSIS = "data_analysis"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    UTILITIES = "utilities"
    BUILD = "build"
    DEVELOPMENT = "development"
    LEGACY = "legacy"

class MigrationStatus(Enum):
    """移行状態"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclass
class ScriptInfo:
    """スクリプト情報"""
    path: Path
    name: str
    language: ScriptLanguage
    category: ScriptCategory
    size_bytes: int
    last_modified: datetime
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    usage_frequency: int = 0
    migration_status: MigrationStatus = MigrationStatus.PENDING
    migration_target: Optional[Path] = None
    migration_notes: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None

@dataclass
class MigrationPlan:
    """移行計画"""
    source_script: ScriptInfo
    target_location: Path
    migration_type: str  # move, copy, refactor, merge, deprecate
    priority: int  # 1-5, 1 = highest
    estimated_effort: str  # low, medium, high
    dependencies: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)

class ScriptMigrationManager:
    """統合スクリプト移行管理システム"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.scripts_inventory: Dict[str, ScriptInfo] = {}
        self.migration_plans: List[MigrationPlan] = []
        self.migration_results: Dict[str, Dict] = {}
        
        # 移行先ディレクトリ定義
        self.target_structure = {
            ScriptCategory.ML_TRAINING: Path("backend/ml/scripts/training"),
            ScriptCategory.DATA_PROCESSING: Path("backend/data/scripts/processing"),
            ScriptCategory.DATA_SYNC: Path("backend/data/scripts/sync"),
            ScriptCategory.DATA_ANALYSIS: Path("backend/data/scripts/analysis"),
            ScriptCategory.DEPLOYMENT: Path("backend/ml/scripts/deployment"),
            ScriptCategory.TESTING: Path("backend/ml/scripts/testing"),
            ScriptCategory.MAINTENANCE: Path("backend/scripts/maintenance"),
            ScriptCategory.UTILITIES: Path("backend/utils"),
            ScriptCategory.BUILD: Path("scripts/build"),
            ScriptCategory.DEVELOPMENT: Path("scripts/development"),
            ScriptCategory.LEGACY: Path("backend/scripts/legacy")
        }
        
        logger.info(f"Script migration manager initialized for {self.project_root}")
    
    def audit_all_scripts(self) -> Dict[str, Any]:
        """全スクリプトの監査実行"""
        logger.info("Starting comprehensive script audit...")
        
        # 監査対象ディレクトリ
        audit_paths = [
            "scripts",
            "backend/scripts", 
            "data/scripts",
            "backend/ml/scripts"
        ]
        
        total_scripts = 0
        for audit_path in audit_paths:
            full_path = self.project_root / audit_path
            if full_path.exists():
                scripts_found = self._scan_directory(full_path)
                total_scripts += len(scripts_found)
                logger.info(f"Found {len(scripts_found)} scripts in {audit_path}")
        
        # 重複検出
        self._detect_duplicates()
        
        # 監査結果生成
        audit_results = self._generate_audit_report()
        
        logger.info(f"Script audit completed: {total_scripts} scripts analyzed")
        return audit_results
    
    def _scan_directory(self, directory: Path) -> List[ScriptInfo]:
        """ディレクトリ内のスクリプトスキャン"""
        scripts = []
        
        # スクリプトファイル拡張子
        script_extensions = {
            ".py": ScriptLanguage.PYTHON,
            ".js": ScriptLanguage.JAVASCRIPT,
            ".ts": ScriptLanguage.TYPESCRIPT,
            ".sh": ScriptLanguage.SHELL,
            ".bash": ScriptLanguage.SHELL
        }
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in script_extensions:
                try:
                    script_info = self._analyze_script(file_path)
                    scripts.append(script_info)
                    self.scripts_inventory[str(file_path)] = script_info
                except Exception as e:
                    logger.warning(f"Failed to analyze script {file_path}: {e}")
        
        return scripts
    
    def _analyze_script(self, file_path: Path) -> ScriptInfo:
        """個別スクリプトの分析"""
        stat = file_path.stat()
        
        # 言語検出
        language = self._detect_language(file_path)
        
        # カテゴリ推定
        category = self._categorize_script(file_path)
        
        # 依存関係分析
        dependencies = self._extract_dependencies(file_path, language)
        
        # 説明抽出
        description = self._extract_description(file_path, language)
        
        return ScriptInfo(
            path=file_path,
            name=file_path.name,
            language=language,
            category=category,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            dependencies=dependencies,
            description=description
        )
    
    def _detect_language(self, file_path: Path) -> ScriptLanguage:
        """スクリプト言語検出"""
        suffix = file_path.suffix.lower()
        
        if suffix == ".py":
            return ScriptLanguage.PYTHON
        elif suffix == ".js":
            return ScriptLanguage.JAVASCRIPT
        elif suffix == ".ts":
            return ScriptLanguage.TYPESCRIPT
        elif suffix in [".sh", ".bash"]:
            return ScriptLanguage.SHELL
        else:
            return ScriptLanguage.UNKNOWN
    
    def _categorize_script(self, file_path: Path) -> ScriptCategory:
        """スクリプトカテゴリ推定"""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        # パスベースの分類
        if "train" in path_str or "training" in path_str:
            return ScriptCategory.ML_TRAINING
        elif "sync" in path_str or "dmm" in path_str or "api" in path_str:
            return ScriptCategory.DATA_SYNC
        elif "analysis" in path_str or "analyze" in path_str:
            return ScriptCategory.DATA_ANALYSIS
        elif "deploy" in path_str or "deployment" in path_str:
            return ScriptCategory.DEPLOYMENT
        elif "test" in path_str or "testing" in path_str:
            return ScriptCategory.TESTING
        elif "maintenance" in path_str or "diagnose" in path_str or "check" in path_str:
            return ScriptCategory.MAINTENANCE
        elif "build" in path_str:
            return ScriptCategory.BUILD
        elif "development" in path_str or "dev" in path_str:
            return ScriptCategory.DEVELOPMENT
        elif "processing" in path_str or "process" in path_str:
            return ScriptCategory.DATA_PROCESSING
        elif "util" in path_str or "helper" in path_str:
            return ScriptCategory.UTILITIES
        
        # ファイル名ベースの分類
        if "train" in name:
            return ScriptCategory.ML_TRAINING
        elif "sync" in name or "dmm" in name:
            return ScriptCategory.DATA_SYNC
        elif "deploy" in name:
            return ScriptCategory.DEPLOYMENT
        elif "test" in name:
            return ScriptCategory.TESTING
        else:
            return ScriptCategory.UTILITIES
    
    def _extract_dependencies(self, file_path: Path, language: ScriptLanguage) -> List[str]:
        """依存関係抽出"""
        dependencies = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if language == ScriptLanguage.PYTHON:
                # Python import文の抽出
                import_patterns = [
                    r'^import\s+([^\s]+)',
                    r'^from\s+([^\s]+)\s+import',
                ]
                for pattern in import_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    dependencies.extend(matches)
            
            elif language in [ScriptLanguage.JAVASCRIPT, ScriptLanguage.TYPESCRIPT]:
                # JavaScript/TypeScript import文の抽出
                import_patterns = [
                    r"import.*from\s+['\"]([^'\"]+)['\"]",
                    r"require\(['\"]([^'\"]+)['\"]\)",
                ]
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    dependencies.extend(matches)
        
        except Exception as e:
            logger.warning(f"Failed to extract dependencies from {file_path}: {e}")
        
        return list(set(dependencies))  # 重複除去
    
    def _extract_description(self, file_path: Path, language: ScriptLanguage) -> str:
        """スクリプトの説明抽出"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if language == ScriptLanguage.PYTHON:
                # Pythonドキュメント文字列の抽出
                try:
                    tree = ast.parse(content)
                    if (tree.body and isinstance(tree.body[0], ast.Expr) 
                        and isinstance(tree.body[0].value, ast.Constant)):
                        return tree.body[0].value.value
                except:
                    pass
                
                # コメントから抽出
                lines = content.split('\n')
                for line in lines[:10]:  # 最初の10行をチェック
                    line = line.strip()
                    if line.startswith('"""') or line.startswith("'''"):
                        return line.strip('"""\'').strip()
                    elif line.startswith('#'):
                        return line[1:].strip()
            
            elif language in [ScriptLanguage.JAVASCRIPT, ScriptLanguage.TYPESCRIPT]:
                # JavaScript/TypeScriptコメントから抽出
                lines = content.split('\n')
                for line in lines[:10]:
                    line = line.strip()
                    if line.startswith('/**') or line.startswith('//'):
                        return line.strip('/*/ ').strip()
        
        except Exception as e:
            logger.warning(f"Failed to extract description from {file_path}: {e}")
        
        return ""
    
    def _detect_duplicates(self):
        """重複スクリプト検出"""
        logger.info("Detecting duplicate scripts...")
        
        # ファイル名ベースの重複検出
        name_groups = {}
        for path, script_info in self.scripts_inventory.items():
            name = script_info.name
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(script_info)
        
        # 重複をマーク
        for name, scripts in name_groups.items():
            if len(scripts) > 1:
                # 最新のものを主とし、他を重複とする
                scripts.sort(key=lambda x: x.last_modified, reverse=True)
                primary = scripts[0]
                
                for duplicate in scripts[1:]:
                    duplicate.is_duplicate = True
                    duplicate.duplicate_of = primary.path
                    logger.info(f"Marked {duplicate.path} as duplicate of {primary.path}")
    
    def generate_migration_plans(self) -> List[MigrationPlan]:
        """移行計画生成"""
        logger.info("Generating migration plans...")
        
        self.migration_plans = []
        
        for path, script_info in self.scripts_inventory.items():
            if script_info.is_duplicate:
                # 重複ファイルは削除計画
                plan = MigrationPlan(
                    source_script=script_info,
                    target_location=self.project_root / "backend/scripts/legacy" / script_info.name,
                    migration_type="deprecate",
                    priority=5,
                    estimated_effort="low"
                )
            else:
                # カテゴリに基づく移行計画
                target_dir = self.target_structure.get(script_info.category)
                if target_dir:
                    target_path = self.project_root / target_dir / script_info.name
                    
                    # 移行タイプ決定
                    if script_info.category in [ScriptCategory.ML_TRAINING, ScriptCategory.DATA_PROCESSING]:
                        migration_type = "move"
                        priority = 1
                    elif script_info.category in [ScriptCategory.TESTING, ScriptCategory.UTILITIES]:
                        migration_type = "copy"
                        priority = 2
                    else:
                        migration_type = "move"
                        priority = 3
                    
                    plan = MigrationPlan(
                        source_script=script_info,
                        target_location=target_path,
                        migration_type=migration_type,
                        priority=priority,
                        estimated_effort="medium",
                        validation_steps=[
                            "Verify dependencies are available",
                            "Test script execution",
                            "Update documentation"
                        ]
                    )
                else:
                    # その他は現在の場所に残す
                    continue
            
            self.migration_plans.append(plan)
        
        # 優先度順にソート
        self.migration_plans.sort(key=lambda x: x.priority)
        
        logger.info(f"Generated {len(self.migration_plans)} migration plans")
        return self.migration_plans
    
    def execute_migration_plan(self, plan: MigrationPlan) -> Dict[str, Any]:
        """移行計画実行"""
        logger.info(f"Executing migration plan for {plan.source_script.name}")
        
        result = {
            "script_name": plan.source_script.name,
            "migration_type": plan.migration_type,
            "start_time": datetime.now(),
            "status": "started",
            "error": None
        }
        
        try:
            # ターゲットディレクトリ作成
            plan.target_location.parent.mkdir(parents=True, exist_ok=True)
            
            if plan.migration_type == "move":
                shutil.move(str(plan.source_script.path), str(plan.target_location))
                logger.info(f"Moved {plan.source_script.path} to {plan.target_location}")
            
            elif plan.migration_type == "copy":
                shutil.copy2(str(plan.source_script.path), str(plan.target_location))
                logger.info(f"Copied {plan.source_script.path} to {plan.target_location}")
            
            elif plan.migration_type == "deprecate":
                # 重複ファイルの削除
                plan.source_script.path.unlink()
                logger.info(f"Deprecated (deleted) {plan.source_script.path}")
            
            # 移行状態更新
            plan.source_script.migration_status = MigrationStatus.COMPLETED
            plan.source_script.migration_target = plan.target_location
            
            result["status"] = "completed"
            result["end_time"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Migration failed for {plan.source_script.name}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            plan.source_script.migration_status = MigrationStatus.FAILED
        
        return result
    
    def execute_all_migrations(self) -> Dict[str, Any]:
        """全移行計画の実行"""
        logger.info("Executing all migration plans...")
        
        results = {
            "total_plans": len(self.migration_plans),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        for plan in self.migration_plans:
            try:
                result = self.execute_migration_plan(plan)
                results["details"].append(result)
                
                if result["status"] == "completed":
                    results["completed"] += 1
                elif result["status"] == "failed":
                    results["failed"] += 1
                else:
                    results["skipped"] += 1
            
            except Exception as e:
                logger.error(f"Failed to execute migration plan: {e}")
                results["failed"] += 1
        
        logger.info(f"Migration complete: {results['completed']} completed, {results['failed']} failed")
        return results
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """監査レポート生成"""
        total_scripts = len(self.scripts_inventory)
        duplicates = sum(1 for s in self.scripts_inventory.values() if s.is_duplicate)
        
        # カテゴリ別統計
        category_stats = {}
        for script_info in self.scripts_inventory.values():
            category = script_info.category.value
            if category not in category_stats:
                category_stats[category] = 0
            category_stats[category] += 1
        
        # 言語別統計
        language_stats = {}
        for script_info in self.scripts_inventory.values():
            language = script_info.language.value
            if language not in language_stats:
                language_stats[language] = 0
            language_stats[language] += 1
        
        return {
            "audit_timestamp": datetime.now().isoformat(),
            "total_scripts": total_scripts,
            "duplicates_found": duplicates,
            "unique_scripts": total_scripts - duplicates,
            "category_distribution": category_stats,
            "language_distribution": language_stats,
            "scripts_details": [
                {
                    "path": str(s.path),
                    "name": s.name,
                    "category": s.category.value,
                    "language": s.language.value,
                    "size_kb": round(s.size_bytes / 1024, 2),
                    "is_duplicate": s.is_duplicate,
                    "description": s.description[:100] + "..." if len(s.description) > 100 else s.description
                }
                for s in self.scripts_inventory.values()
            ]
        }
    
    def export_migration_report(self, output_path: Path):
        """移行レポートのエクスポート"""
        audit_results = self._generate_audit_report()
        
        report = {
            "audit_results": audit_results,
            "migration_plans": [
                {
                    "source_path": str(plan.source_script.path),
                    "target_path": str(plan.target_location),
                    "migration_type": plan.migration_type,
                    "priority": plan.priority,
                    "estimated_effort": plan.estimated_effort,
                    "status": plan.source_script.migration_status.value
                }
                for plan in self.migration_plans
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Migration report exported to {output_path}")

# CLI インターフェース
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Script Migration Manager")
        parser.add_argument("--audit", action="store_true", help="Run script audit")
        parser.add_argument("--plan", action="store_true", help="Generate migration plans")
        parser.add_argument("--migrate", action="store_true", help="Execute migrations")
        parser.add_argument("--report", type=str, help="Export report to file")
        
        args = parser.parse_args()
        
        manager = ScriptMigrationManager()
        
        if args.audit:
            audit_results = manager.audit_all_scripts()
            print(f"Audit completed: {audit_results['total_scripts']} scripts found")
        
        if args.plan:
            plans = manager.generate_migration_plans()
            print(f"Generated {len(plans)} migration plans")
        
        if args.migrate:
            results = manager.execute_all_migrations()
            print(f"Migration results: {results['completed']} completed, {results['failed']} failed")
        
        if args.report:
            manager.export_migration_report(Path(args.report))
            print(f"Report exported to {args.report}")
    
    import asyncio
    asyncio.run(main())
#!/usr/bin/env python3
"""
Unified Script Runner for Adult Matching Backend
Provides centralized script execution, management, and monitoring.
"""

import sys
import os
import argparse
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import importlib.util

# Add backend to Python path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScriptRunner:
    """Unified script runner for all backend scripts"""
    
    def __init__(self):
        self.backend_root = BACKEND_ROOT
        self.script_registry = self._build_script_registry()
    
    def _build_script_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of all available scripts"""
        registry = {
            # ML Scripts (now in backend/ml)
            "ml": {
                "training": {
                    "train_768_two_tower": {
                        "path": "backend/ml/scripts/training/train_768_dim_two_tower.py",
                        "description": "Train 768-dimension Two-Tower model",
                        "type": "python"
                    },
                    "train_production_two_tower": {
                        "path": "backend/ml/scripts/training/train_production_two_tower.py", 
                        "description": "Train production-ready Two-Tower model",
                        "type": "python"
                    },
                    "train_standard_model": {
                        "path": "backend/ml/scripts/training/train_standard_model.py",
                        "description": "Train standard recommendation model",
                        "type": "python"
                    },
                    "train_comprehensive": {
                        "path": "backend/ml/scripts/training/train_two_tower_comprehensive.py",
                        "description": "Comprehensive Two-Tower training with full pipeline",
                        "type": "python"
                    },
                    "train_basic": {
                        "path": "backend/ml/scripts/training/train_two_tower_model.py",
                        "description": "Basic Two-Tower model training",
                        "type": "python"
                    }
                },
                "testing": {
                    "test_simple": {
                        "path": "backend/ml/scripts/testing/simple_two_tower_test.py",
                        "description": "Simple Two-Tower model testing",
                        "type": "python"
                    },
                    "test_pgvector": {
                        "path": "backend/ml/scripts/testing/test_768_pgvector_integration.py",
                        "description": "Test 768-dim pgvector integration",
                        "type": "python"
                    },
                    "test_components": {
                        "path": "backend/ml/scripts/testing/test_training_components.py",
                        "description": "Test ML training components",
                        "type": "python"
                    },
                    "verify_768_model": {
                        "path": "backend/ml/scripts/testing/verify_768_model.py",
                        "description": "Verify 768-dimension model functionality",
                        "type": "python"
                    }
                },
                "deployment": {
                    "deploy_model": {
                        "path": "backend/ml/scripts/deployment/model_deployment.py",
                        "description": "Deploy trained models to production",
                        "type": "python"
                    },
                    "standardize_768": {
                        "path": "backend/ml/scripts/standardize_models_768.py",
                        "description": "Standardize models to 768 dimensions",
                        "type": "python"
                    }
                }
            },
            
            # Data Scripts (migrated to backend/data)
            "data": {
                "sync": {
                    "dmm_sync": {
                        "path": "data/scripts/sync/real_dmm_sync.js",
                        "description": "Real DMM API data synchronization",
                        "type": "javascript"
                    },
                    "bulk_sync": {
                        "path": "data/scripts/sync/efficient_dmm_bulk_sync.js",
                        "description": "Efficient bulk DMM data sync",
                        "type": "javascript"
                    },
                    "mega_sync": {
                        "path": "data/scripts/sync/mega_dmm_sync_200k.js",
                        "description": "Mega sync for 200k+ records",
                        "type": "javascript"
                    },
                    "multi_sort_sync": {
                        "path": "data/scripts/sync/multi_sort_dmm_sync.js",
                        "description": "Multi-sort DMM data sync",
                        "type": "javascript"
                    },
                    "test_sync": {
                        "path": "data/scripts/sync/test_dmm_sync_small.js",
                        "description": "Test DMM sync with small dataset",
                        "type": "javascript"
                    }
                },
                "analysis": {
                    "analyze_dmm": {
                        "path": "data/scripts/analysis/analyze_dmm_data.js",
                        "description": "Analyze DMM data quality and completeness",
                        "type": "javascript"
                    },
                    "analyze_reviews": {
                        "path": "data/scripts/analysis/analyze_review_dates.js",
                        "description": "Analyze review date patterns",
                        "type": "javascript"
                    },
                    "content_linking": {
                        "path": "data/scripts/analysis/content_id_linking.js",
                        "description": "Link content IDs across datasets",
                        "type": "javascript"
                    },
                    "accurate_linking": {
                        "path": "data/scripts/analysis/accurate_content_linking.js",
                        "description": "Accurate content ID linking algorithm",
                        "type": "javascript"
                    }
                },
                "maintenance": {
                    "db_check": {
                        "path": "data/scripts/maintenance/quick_db_check.js",
                        "description": "Quick database health check",
                        "type": "javascript"
                    },
                    "diagnose_db": {
                        "path": "data/scripts/maintenance/diagnose_database_issue.js",
                        "description": "Diagnose database issues",
                        "type": "javascript"
                    }
                },
                "processing": {
                    "batch_processor": {
                        "path": "backend/data/scripts/processing/batch_data_processor.py",
                        "description": "Batch data processing pipeline",
                        "type": "python"
                    },
                    "streaming_processor": {
                        "path": "backend/data/scripts/processing/streaming_processor.py",
                        "description": "Real-time streaming data processor",
                        "type": "python"
                    },
                    "validation_runner": {
                        "path": "backend/data/scripts/processing/validation_runner.py",
                        "description": "Data validation pipeline runner",
                        "type": "python"
                    }
                }
            },
            
            # Management Scripts
            "management": {
                "migration": {
                    "script_migration": {
                        "path": "backend/scripts/script_migration_manager.py",
                        "description": "Script migration and organization tool",
                        "type": "python"
                    }
                },
                "monitoring": {
                    "pipeline_monitor": {
                        "path": "backend/scripts/pipeline_monitor.py",
                        "description": "Monitor data and ML pipelines",
                        "type": "python"
                    },
                    "performance_monitor": {
                        "path": "backend/scripts/performance_monitor.py", 
                        "description": "System performance monitoring",
                        "type": "python"
                    }
                },
                "utilities": {
                    "script_runner": {
                        "path": "backend/scripts/unified_script_runner.py",
                        "description": "Unified script execution interface",
                        "type": "python"
                    },
                    "script_manager": {
                        "path": "backend/scripts/utilities/script_manager.py",
                        "description": "Advanced script execution and monitoring",
                        "type": "python"
                    },
                    "migration_manager": {
                        "path": "backend/scripts/script_migration_manager.py",
                        "description": "Script migration and organization tool",
                        "type": "python"
                    }
                }
            },
            
            # Frontend Development Scripts
            "frontend": {
                "development": {
                    "install": {
                        "path": "scripts/development/install.js",
                        "description": "Development environment installation",
                        "type": "javascript"
                    },
                    "release_notes": {
                        "path": "scripts/development/release-notes.js",
                        "description": "Generate release notes",
                        "type": "javascript"
                    },
                    "release_channel": {
                        "path": "scripts/development/release-channel.js",
                        "description": "Manage release channels",
                        "type": "javascript"
                    }
                },
                "build": {
                    "generate_types": {
                        "path": "scripts/build/generate-types.js",
                        "description": "Generate TypeScript type definitions",
                        "type": "javascript"
                    },
                    "build": {
                        "path": "scripts/build/build.js",
                        "description": "Main build process",
                        "type": "javascript"
                    },
                    "bundle": {
                        "path": "scripts/build/bundle.js",
                        "description": "Bundle application resources",
                        "type": "javascript"
                    },
                    "create_plugin_list": {
                        "path": "scripts/build/create-plugin-list.js",
                        "description": "Generate plugin list",
                        "type": "javascript"
                    },
                    "compile_dots": {
                        "path": "scripts/build/compile-dots.js",
                        "description": "Compile Dot template files",
                        "type": "javascript"
                    }
                },
                "utilities": {
                    "utils": {
                        "path": "scripts/utilities/utils.js",
                        "description": "JavaScript general utilities",
                        "type": "javascript"
                    },
                    "ast_grep": {
                        "path": "scripts/utilities/ast_grep.js",
                        "description": "AST-based code search tool",
                        "type": "javascript"
                    },
                    "type_utils": {
                        "path": "scripts/utilities/type-utils.js",
                        "description": "TypeScript type utilities",
                        "type": "javascript"
                    }
                }
            }
        }
        
        return registry
    
    def list_scripts(self, category: Optional[str] = None) -> None:
        """List all available scripts"""
        if category and category in self.script_registry:
            self._print_category(category, self.script_registry[category])
        else:
            for cat_name, cat_scripts in self.script_registry.items():
                self._print_category(cat_name, cat_scripts)
                print()
    
    def _print_category(self, category: str, scripts: Dict) -> None:
        """Print scripts in a category"""
        print(f"📁 {category.upper()} Scripts:")
        for subcategory, subscripts in scripts.items():
            print(f"  📂 {subcategory}:")
            for script_name, script_info in subscripts.items():
                print(f"    • {script_name}: {script_info['description']}")
    
    def run_script(self, script_path: str, *args) -> int:
        """Run a script by path or registry name"""
        # Check if it's a registry name
        script_info = self._find_script_in_registry(script_path)
        
        if script_info:
            actual_path = self.backend_root / script_info["path"]
            script_type = script_info["type"]
        else:
            # Assume it's a direct path
            actual_path = Path(script_path)
            if not actual_path.is_absolute():
                actual_path = self.backend_root / actual_path
            
            # Determine script type from extension
            if actual_path.suffix == ".py":
                script_type = "python"
            elif actual_path.suffix == ".js":
                script_type = "javascript"
            else:
                logger.error(f"Unsupported script type: {actual_path.suffix}")
                return 1
        
        if not actual_path.exists():
            logger.error(f"Script not found: {actual_path}")
            return 1
        
        # Execute the script
        logger.info(f"Executing {script_type} script: {actual_path}")
        
        try:
            if script_type == "python":
                return self._run_python_script(actual_path, *args)
            elif script_type == "javascript":
                return self._run_javascript_script(actual_path, *args)
            else:
                logger.error(f"Unsupported script type: {script_type}")
                return 1
                
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return 1
    
    def _find_script_in_registry(self, script_name: str) -> Optional[Dict]:
        """Find script info in registry by name"""
        for category in self.script_registry.values():
            for subcategory in category.values():
                if script_name in subcategory:
                    return subcategory[script_name]
        return None
    
    def _run_python_script(self, script_path: Path, *args) -> int:
        """Run a Python script"""
        cmd = [sys.executable, str(script_path)] + list(args)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.backend_root)
        
        result = subprocess.run(cmd, env=env, cwd=self.backend_root)
        return result.returncode
    
    def _run_javascript_script(self, script_path: Path, *args) -> int:
        """Run a JavaScript script"""
        cmd = ["node", str(script_path)] + list(args)
        
        result = subprocess.run(cmd, cwd=self.backend_root)
        return result.returncode
    
    def run_pipeline(self, pipeline_name: str) -> int:
        """Run a predefined pipeline of scripts"""
        pipelines = {
            "ml_full": [
                "verify_768_model",
                "train_768_two_tower", 
                "train_production_two_tower",
                "standardize_768"
            ],
            "data_sync": [
                "dmm_sync",
                "analyze_dmm",
                "db_check"
            ],
            "full_build": [
                "generate_types",
                "build",
                "bundle"
            ],
            "data_quality": [
                "analyze_dmm",
                "diagnose_db",
                "validation_runner"
            ]
        }
        
        if pipeline_name not in pipelines:
            logger.error(f"Pipeline '{pipeline_name}' not found. Available: {list(pipelines.keys())}")
            return 1
            
        pipeline_scripts = pipelines[pipeline_name]
        logger.info(f"🚀 Running pipeline '{pipeline_name}' with {len(pipeline_scripts)} scripts")
        
        for i, script_name in enumerate(pipeline_scripts, 1):
            logger.info(f"📋 Pipeline step {i}/{len(pipeline_scripts)}: {script_name}")
            result = self.run_script(script_name)
            
            if result != 0:
                logger.error(f"❌ Pipeline '{pipeline_name}' failed at step {i}: {script_name}")
                return result
                
        logger.info(f"✅ Pipeline '{pipeline_name}' completed successfully")
        return 0
    
    def list_pipelines(self) -> None:
        """List available pipelines"""
        pipelines = {
            "ml_full": "Complete ML training pipeline (verify → train 768 → train production → standardize)",
            "data_sync": "Data synchronization pipeline (sync → analyze → health check)",
            "full_build": "Frontend build pipeline (types → build → bundle)",
            "data_quality": "Data quality assessment pipeline (analyze → diagnose → validate)"
        }
        
        print("🔧 Available Pipelines:")
        for name, description in pipelines.items():
            print(f"  • {name}: {description}")
    
    def search_scripts(self, query: str) -> None:
        """Search for scripts by name or description"""
        print(f"🔍 Searching for scripts matching: '{query}'")
        found = False
        
        for cat_name, category in self.script_registry.items():
            for subcat_name, subcategory in category.items():
                for script_name, script_info in subcategory.items():
                    if (query.lower() in script_name.lower() or 
                        query.lower() in script_info["description"].lower()):
                        if not found:
                            print("Found scripts:")
                            found = True
                        print(f"  📁 {cat_name}/{subcat_name}/{script_name}")
                        print(f"     {script_info['description']}")
                        print(f"     Path: {script_info['path']}")
                        print()
        
        if not found:
            print("No scripts found matching the query.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Script Runner for Adult Matching Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_script_runner.py list                    # List all scripts
  python unified_script_runner.py list ml                 # List ML scripts only
  python unified_script_runner.py search dmm              # Search for DMM-related scripts
  python unified_script_runner.py run train_768_two_tower # Run training script
  python unified_script_runner.py pipeline ml_full        # Run ML training pipeline
  python unified_script_runner.py pipelines               # List available pipelines
  python unified_script_runner.py run data/sync/dmm_sync.js --batch-size 100
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available scripts")
    list_parser.add_argument("category", nargs="?", help="Category to list (optional)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for scripts")
    search_parser.add_argument("query", help="Search query")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a script")
    run_parser.add_argument("script", help="Script name or path")
    run_parser.add_argument("args", nargs="*", help="Script arguments")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run a pipeline")
    pipeline_parser.add_argument("pipeline", help="Pipeline name")
    
    # List pipelines command
    pipelines_parser = subparsers.add_parser("pipelines", help="List available pipelines")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    runner = ScriptRunner()
    
    if args.command == "list":
        runner.list_scripts(args.category)
    elif args.command == "search":
        runner.search_scripts(args.query)
    elif args.command == "run":
        return runner.run_script(args.script, *args.args)
    elif args.command == "pipeline":
        return runner.run_pipeline(args.pipeline)
    elif args.command == "pipelines":
        runner.list_pipelines()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
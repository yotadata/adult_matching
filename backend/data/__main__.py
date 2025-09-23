"""
Data Package Entry Point

データパッケージコマンドライン実行インターフェース
- 全データ操作のエントリーポイント
- コマンドライン引数での実行制御
- システム管理とデバッグ機能
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from backend.ml.utils.logger import get_ml_logger
from backend.data.unified_data_manager import UnifiedDataManager, create_unified_data_manager
from backend.data.config_manager import get_config_manager, get_config
from backend.data.pipelines.pipeline_manager import PipelineConfig, PipelineStage

# スクリプト統合機能の遅延インポート
try:
    from backend.data.script_integration import create_integrated_script_manager
    SCRIPT_INTEGRATION_AVAILABLE = True
except ImportError:
    SCRIPT_INTEGRATION_AVAILABLE = False

logger = get_ml_logger(__name__)

def create_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーの作成"""
    
    parser = argparse.ArgumentParser(
        description="Adult Matching Data Package - データ管理システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 完全データフロー実行
  python -m backend.data run-flow --sources api,scraping --format json,csv
  
  # パイプライン実行
  python -m backend.data run-pipeline full_data_pipeline
  
  # システム状態確認
  python -m backend.data status
  
  # ヘルスチェック実行
  python -m backend.data health-check
  
  # 設定管理
  python -m backend.data config --show
  python -m backend.data config --update api.rate_limit_per_second=0.5
  
  # データクリーンアップ
  python -m backend.data cleanup --retention-days 30
  
  # パフォーマンステスト
  python -m backend.data test-performance
        """
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production", "testing"],
        help="実行環境指定"
    )
    
    parser.add_argument(
        "--config-file", "-c",
        type=Path,
        help="設定ファイルパス"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")
    
    # run-flow コマンド
    flow_parser = subparsers.add_parser("run-flow", help="完全データフロー実行")
    flow_parser.add_argument(
        "--sources",
        default="api,scraping",
        help="データソース (api,scraping,file)"
    )
    flow_parser.add_argument(
        "--format",
        default="json",
        help="出力形式 (json,csv,parquet)"
    )
    flow_parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.8,
        help="品質しきい値 (0.0-1.0)"
    )
    
    # run-pipeline コマンド
    pipeline_parser = subparsers.add_parser("run-pipeline", help="パイプライン実行")
    pipeline_parser.add_argument("pipeline_name", help="パイプライン名")
    pipeline_parser.add_argument(
        "--input-data",
        type=Path,
        help="入力データファイル"
    )
    
    # create-pipeline コマンド
    create_parser = subparsers.add_parser("create-pipeline", help="パイプライン作成")
    create_parser.add_argument("name", help="パイプライン名")
    create_parser.add_argument(
        "--stages",
        required=True,
        help="ステージ (ingestion,processing,validation,export,ml_training)"
    )
    create_parser.add_argument("--schedule", help="スケジュール (daily,hourly)")
    create_parser.add_argument("--sources", default="api", help="データソース")
    create_parser.add_argument("--format", default="json", help="出力形式")
    
    # status コマンド
    subparsers.add_parser("status", help="システム状態確認")
    
    # health-check コマンド
    subparsers.add_parser("health-check", help="ヘルスチェック実行")
    
    # config コマンド
    config_parser = subparsers.add_parser("config", help="設定管理")
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--show", action="store_true", help="設定表示")
    config_group.add_argument("--summary", action="store_true", help="設定サマリー")
    config_group.add_argument("--validate", action="store_true", help="設定検証")
    config_group.add_argument("--export", action="store_true", help="設定エクスポート")
    config_group.add_argument("--update", help="設定更新 (section.key=value)")
    config_group.add_argument("--reset", help="設定リセット (section名)")
    
    # cleanup コマンド
    cleanup_parser = subparsers.add_parser("cleanup", help="データクリーンアップ")
    cleanup_parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="保持日数"
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行内容のプレビュー"
    )
    
    # list-pipelines コマンド
    subparsers.add_parser("list-pipelines", help="パイプライン一覧")
    
    # dashboard コマンド
    subparsers.add_parser("dashboard", help="ダッシュボードデータ取得")
    
    # test-performance コマンド
    test_parser = subparsers.add_parser("test-performance", help="パフォーマンステスト")
    test_parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="テスト時間(秒)"
    )
    test_parser.add_argument(
        "--concurrent-jobs",
        type=int,
        default=3,
        help="同時実行ジョブ数"
    )
    
    return parser

async def run_data_flow_command(args, data_manager: UnifiedDataManager):
    """データフロー実行コマンド"""
    sources = args.sources.split(",")
    
    logger.info(f"Starting data flow with sources: {sources}")
    
    result = await data_manager.run_full_data_flow(
        sources=sources,
        target_format=args.format,
        quality_threshold=args.quality_threshold
    )
    
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))

async def run_pipeline_command(args, data_manager: UnifiedDataManager):
    """パイプライン実行コマンド"""
    logger.info(f"Running pipeline: {args.pipeline_name}")
    
    result = await data_manager.pipeline_manager.run_pipeline(
        args.pipeline_name,
        input_data=args.input_data
    )
    
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.status.value,
        "processing_time_seconds": result.processing_time_seconds,
        "input_records": result.input_records,
        "output_records": result.output_records,
        "quality_score": result.quality_score
    }, indent=2, default=str, ensure_ascii=False))

def create_pipeline_command(args, data_manager: UnifiedDataManager):
    """パイプライン作成コマンド"""
    
    # ステージの変換
    stage_mapping = {
        "ingestion": PipelineStage.INGESTION,
        "processing": PipelineStage.PROCESSING,
        "validation": PipelineStage.VALIDATION,
        "export": PipelineStage.EXPORT,
        "ml_training": PipelineStage.ML_TRAINING
    }
    
    stages = [
        stage_mapping[stage.strip()]
        for stage in args.stages.split(",")
        if stage.strip() in stage_mapping
    ]
    
    if not stages:
        print("エラー: 有効なステージが指定されていません")
        return
    
    # パイプライン設定作成
    config = PipelineConfig(
        name=args.name,
        stages=stages,
        schedule=args.schedule,
        data_sources=args.sources.split(","),
        output_format=args.format
    )
    
    # パイプライン登録
    pipeline_id = data_manager.pipeline_manager.register_pipeline(config)
    
    print(f"パイプライン作成完了: {pipeline_id}")

async def status_command(args, data_manager: UnifiedDataManager):
    """システム状態確認コマンド"""
    
    # システムメトリクス
    metrics = await data_manager.get_system_metrics()
    
    # データフロー状態
    data_flow = await data_manager.get_data_flow_status()
    
    # コンポーネント状態
    components = data_manager.get_component_status()
    
    # パイプライン状態
    pipeline_status = data_manager.pipeline_manager.get_system_status()
    
    status_report = {
        "timestamp": metrics.timestamp.isoformat(),
        "system_health": {
            "cpu_usage": f"{metrics.cpu_usage:.1f}%",
            "memory_usage": f"{metrics.memory_usage:.1f}%",
            "disk_usage": f"{metrics.disk_usage:.1f}%",
            "uptime_hours": f"{metrics.uptime_hours:.1f}h"
        },
        "data_flow": {
            "ingestion_status": data_flow.ingestion_status,
            "processing_status": data_flow.processing_status,
            "validation_status": data_flow.validation_status,
            "export_status": data_flow.export_status,
            "records_today": data_flow.total_records_today,
            "failed_today": data_flow.failed_records_today
        },
        "pipelines": {
            "registered": pipeline_status["registered_pipelines"],
            "active_runs": pipeline_status["active_runs"],
            "completed_runs": pipeline_status["completed_runs"]
        },
        "components": {
            name: status["status"]
            for name, status in components.items()
        }
    }
    
    print(json.dumps(status_report, indent=2, ensure_ascii=False))

async def health_check_command(args, data_manager: UnifiedDataManager):
    """ヘルスチェックコマンド"""
    
    health_report = await data_manager.run_health_check()
    
    print(json.dumps(health_report, indent=2, ensure_ascii=False))
    
    # 終了コード設定
    if health_report["overall_status"] == "error":
        sys.exit(1)
    elif health_report["overall_status"] == "warning":
        sys.exit(2)

def config_command(args, config_manager):
    """設定管理コマンド"""
    
    if args.show:
        config = config_manager.load_config()
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    
    elif args.summary:
        summary = config_manager.get_config_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    elif args.validate:
        config = config_manager.load_config()
        errors = config_manager.validate_config(config)
        
        if errors:
            print("設定エラー:")
            for section, section_errors in errors.items():
                print(f"  {section}:")
                for error in section_errors:
                    print(f"    - {error}")
            sys.exit(1)
        else:
            print("設定は有効です")
    
    elif args.export:
        exported = config_manager.export_config(include_sensitive=False)
        print(json.dumps(exported, indent=2, ensure_ascii=False))
    
    elif args.update:
        # section.key=value 形式の解析
        try:
            key_path, value = args.update.split("=", 1)
            sections = key_path.split(".")
            
            if len(sections) != 2:
                print("エラー: 更新形式は 'section.key=value' です")
                sys.exit(1)
            
            section, key = sections
            
            # 値の型変換
            try:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)
            except:
                pass  # 文字列のまま
            
            success = config_manager.update_config(section, {key: value})
            
            if success:
                print(f"設定更新完了: {section}.{key} = {value}")
            else:
                print("設定更新に失敗しました")
                sys.exit(1)
                
        except ValueError:
            print("エラー: 更新形式は 'section.key=value' です")
            sys.exit(1)
    
    elif args.reset:
        sections = [args.reset] if args.reset != "all" else None
        success = config_manager.reset_to_defaults(sections)
        
        if success:
            print(f"設定リセット完了: {args.reset}")
        else:
            print("設定リセットに失敗しました")
            sys.exit(1)

async def cleanup_command(args, data_manager: UnifiedDataManager):
    """クリーンアップコマンド"""
    
    if args.dry_run:
        print(f"クリーンアップ対象 (保持日数: {args.retention_days}日):")
        # TODO: ドライラン機能実装
        print("ドライラン機能は今後実装予定です")
    else:
        result = await data_manager.cleanup_old_data(args.retention_days)
        print(json.dumps(result, indent=2, ensure_ascii=False))

def list_pipelines_command(args, data_manager: UnifiedDataManager):
    """パイプライン一覧コマンド"""
    
    pipelines = data_manager.pipeline_manager.list_pipelines()
    
    if not pipelines:
        print("登録されているパイプラインはありません")
    else:
        print("登録済みパイプライン:")
        for pipeline in pipelines:
            print(f"  - {pipeline['name']}")
            print(f"    ステージ: {', '.join(pipeline['stages'])}")
            print(f"    スケジュール: {pipeline['schedule'] or 'なし'}")
            print(f"    状態: {pipeline['status']}")
            print()

async def dashboard_command(args, data_manager: UnifiedDataManager):
    """ダッシュボードデータ取得コマンド"""
    
    dashboard_data = await data_manager.get_dashboard_data()
    print(json.dumps(dashboard_data, indent=2, ensure_ascii=False))

async def test_performance_command(args, data_manager: UnifiedDataManager):
    """パフォーマンステストコマンド"""
    
    print(f"パフォーマンステスト開始 (時間: {args.duration}秒, 同時ジョブ: {args.concurrent_jobs})")
    
    import time
    start_time = time.time()
    
    # 複数ジョブの同時実行
    tasks = []
    for i in range(args.concurrent_jobs):
        task = data_manager.run_full_data_flow(
            sources=["api"],
            target_format="json",
            quality_threshold=0.5
        )
        tasks.append(task)
    
    # 指定時間まで実行
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=args.duration
        )
        
        # 結果分析
        successful_jobs = len([r for r in results if isinstance(r, dict) and r.get("success")])
        failed_jobs = len(results) - successful_jobs
        
        execution_time = time.time() - start_time
        
        performance_report = {
            "test_duration_seconds": execution_time,
            "concurrent_jobs": args.concurrent_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": successful_jobs / len(results) if results else 0,
            "average_job_time": execution_time / len(results) if results else 0
        }
        
        print(json.dumps(performance_report, indent=2, ensure_ascii=False))
        
    except asyncio.TimeoutError:
        print(f"テストタイムアウト ({args.duration}秒)")

async def main():
    """メイン実行関数"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # ログレベル設定
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 設定管理初期化
    try:
        config_manager = get_config_manager()
        
        # 設定コマンドは特別処理
        if args.command == "config":
            config_command(args, config_manager)
            return
        
        # データ管理システム初期化
        data_manager = create_unified_data_manager()
        
        # コマンド実行
        if args.command == "run-flow":
            await run_data_flow_command(args, data_manager)
        
        elif args.command == "run-pipeline":
            await run_pipeline_command(args, data_manager)
        
        elif args.command == "create-pipeline":
            create_pipeline_command(args, data_manager)
        
        elif args.command == "status":
            await status_command(args, data_manager)
        
        elif args.command == "health-check":
            await health_check_command(args, data_manager)
        
        elif args.command == "cleanup":
            await cleanup_command(args, data_manager)
        
        elif args.command == "list-pipelines":
            list_pipelines_command(args, data_manager)
        
        elif args.command == "dashboard":
            await dashboard_command(args, data_manager)
        
        elif args.command == "test-performance":
            await test_performance_command(args, data_manager)
        
        else:
            print(f"不明なコマンド: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n実行がキャンセルされました")
        sys.exit(130)
    
    except Exception as error:
        logger.error(f"実行エラー: {error}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
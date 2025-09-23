#!/usr/bin/env python3
"""
包括的要件検証スクリプト

Backend Refactoring プロジェクトの全要件達成を検証します。
"""

import asyncio
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class VerificationResult:
    requirement_id: str
    title: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'SKIP'
    details: List[str]
    metrics: Dict[str, Any]
    duration_ms: float

class ComprehensiveVerifier:
    """包括的検証システム"""

    def __init__(self):
        self.project_root = project_root
        self.results: List[VerificationResult] = []
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('comprehensive_verifier')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def verify_requirement_1_edge_functions(self) -> VerificationResult:
        """要件1: Edge Functions統合とクリーンアップの検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            # Edge Functions ディレクトリ構造確認
            functions_dir = self.project_root / 'supabase' / 'functions'

            # 1. 統合された関数の存在確認
            expected_functions = [
                'recommendations/enhanced_two_tower',
                'user-management/likes',
                'user-management/embeddings',
                'user-management/account',
                'content/feed'
            ]

            existing_functions = []
            for func_path in expected_functions:
                func_dir = functions_dir / func_path
                if func_dir.exists() and (func_dir / 'index.ts').exists():
                    existing_functions.append(func_path)
                    details.append(f"✅ 統合関数が存在: {func_path}")
                else:
                    details.append(f"❌ 統合関数が不在: {func_path}")
                    status = 'FAIL'

            metrics['integrated_functions_count'] = len(existing_functions)
            metrics['expected_functions_count'] = len(expected_functions)

            # 2. 共有ユーティリティの確認
            shared_dir = functions_dir / '_shared'
            shared_modules = ['auth.ts', 'database.ts', 'validation.ts', 'monitoring.ts']

            existing_shared = []
            for module in shared_modules:
                if (shared_dir / module).exists():
                    existing_shared.append(module)
                    details.append(f"✅ 共有モジュール存在: {module}")
                else:
                    details.append(f"⚠️  共有モジュール不在: {module}")
                    if status == 'PASS':
                        status = 'WARNING'

            metrics['shared_modules_count'] = len(existing_shared)

            # 3. TypeScript ファイル数カウント
            ts_files = list(functions_dir.rglob('*.ts'))
            metrics['total_ts_files'] = len(ts_files)
            details.append(f"📊 TypeScript ファイル総数: {len(ts_files)}")

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="REQ-1",
            title="Edge Functions統合とクリーンアップ",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def verify_requirement_2_ml_pipeline(self) -> VerificationResult:
        """要件2: MLパイプライン構造の最適化検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            ml_dir = self.project_root / 'backend' / 'ml'

            # 1. MLディレクトリ構造確認
            expected_ml_dirs = [
                'models', 'training', 'preprocessing', 'inference', 'evaluation'
            ]

            existing_ml_dirs = []
            for dir_name in expected_ml_dirs:
                if (ml_dir / dir_name).exists():
                    existing_ml_dirs.append(dir_name)
                    details.append(f"✅ MLディレクトリ存在: {dir_name}")
                else:
                    details.append(f"❌ MLディレクトリ不在: {dir_name}")
                    status = 'FAIL'

            metrics['ml_directories_count'] = len(existing_ml_dirs)

            # 2. Pythonファイル数確認
            python_files = list(ml_dir.rglob('*.py'))
            metrics['ml_python_files'] = len(python_files)
            details.append(f"📊 MLパイプライン Python ファイル数: {len(python_files)}")

            # 3. 統合テストファイル確認
            tests_dir = self.project_root / 'backend' / 'tests' / 'integration' / 'ml'
            if tests_dir.exists():
                test_files = list(tests_dir.glob('*.py'))
                metrics['ml_test_files'] = len(test_files)
                details.append(f"✅ ML統合テストファイル数: {len(test_files)}")
            else:
                details.append("❌ ML統合テストディレクトリが不在")
                status = 'FAIL'

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="REQ-2",
            title="MLパイプライン構造の最適化",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def verify_requirement_3_data_pipeline(self) -> VerificationResult:
        """要件3: データ処理パイプライン統合検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            data_dir = self.project_root / 'backend' / 'data'

            # 1. データディレクトリ構造確認
            expected_data_dirs = ['sync', 'processing', 'storage', 'quality']

            existing_data_dirs = []
            for dir_name in expected_data_dirs:
                if (data_dir / dir_name).exists():
                    existing_data_dirs.append(dir_name)
                    details.append(f"✅ データディレクトリ存在: {dir_name}")
                else:
                    details.append(f"❌ データディレクトリ不在: {dir_name}")
                    status = 'FAIL'

            metrics['data_directories_count'] = len(existing_data_dirs)

            # 2. DMM同期スクリプト確認
            dmm_sync_dir = data_dir / 'sync' / 'dmm'
            if dmm_sync_dir.exists():
                dmm_files = list(dmm_sync_dir.glob('*.py'))
                metrics['dmm_sync_files'] = len(dmm_files)
                details.append(f"✅ DMM同期ファイル数: {len(dmm_files)}")
            else:
                details.append("❌ DMM同期ディレクトリが不在")
                status = 'FAIL'

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="REQ-3",
            title="データ処理パイプライン統合",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def verify_requirement_4_directory_structure(self) -> VerificationResult:
        """要件4: ディレクトリ構造の合理化検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            # 1. バックエンド主要ディレクトリ確認
            backend_dir = self.project_root / 'backend'
            expected_backend_dirs = [
                'ml', 'data', 'monitoring', 'optimization', 'tests', 'scripts'
            ]

            existing_backend_dirs = []
            for dir_name in expected_backend_dirs:
                if (backend_dir / dir_name).exists():
                    existing_backend_dirs.append(dir_name)
                    details.append(f"✅ バックエンドディレクトリ存在: {dir_name}")
                else:
                    details.append(f"❌ バックエンドディレクトリ不在: {dir_name}")
                    status = 'FAIL'

            metrics['backend_directories_count'] = len(existing_backend_dirs)

            # 2. 一時ファイル・キャッシュファイルの確認
            temp_patterns = ['*.pyc', '__pycache__', '*.tmp', '.DS_Store']
            temp_files_count = 0

            for pattern in temp_patterns:
                temp_files = list(self.project_root.rglob(pattern))
                # .venv と node_modules は除外
                filtered_files = [
                    f for f in temp_files
                    if '.venv' not in str(f) and 'node_modules' not in str(f)
                ]
                temp_files_count += len(filtered_files)

            metrics['temp_files_count'] = temp_files_count

            if temp_files_count == 0:
                details.append("✅ 一時ファイルがクリーンアップ済み")
            else:
                details.append(f"⚠️  一時ファイルが残存: {temp_files_count}個")
                if status == 'PASS':
                    status = 'WARNING'

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="REQ-4",
            title="ディレクトリ構造の合理化",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def verify_requirement_5_testing(self) -> VerificationResult:
        """要件5: 包括的テスティング統合検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            tests_dir = self.project_root / 'backend' / 'tests'

            # 1. テストディレクトリ構造確認
            expected_test_dirs = ['unit', 'integration', 'e2e']

            existing_test_dirs = []
            for dir_name in expected_test_dirs:
                if (tests_dir / dir_name).exists():
                    existing_test_dirs.append(dir_name)
                    details.append(f"✅ テストディレクトリ存在: {dir_name}")
                else:
                    details.append(f"❌ テストディレクトリ不在: {dir_name}")
                    status = 'FAIL'

            metrics['test_directories_count'] = len(existing_test_dirs)

            # 2. テストファイル数確認
            total_test_files = 0
            for test_type in expected_test_dirs:
                test_type_dir = tests_dir / test_type
                if test_type_dir.exists():
                    test_files = list(test_type_dir.rglob('test_*.py'))
                    metrics[f'{test_type}_test_files'] = len(test_files)
                    total_test_files += len(test_files)
                    details.append(f"📊 {test_type}テストファイル数: {len(test_files)}")

            metrics['total_test_files'] = total_test_files

            # 3. 統合コンテンツAPIテスト確認
            content_tests_dir = tests_dir / 'integration' / 'content'
            if content_tests_dir.exists():
                content_test_files = list(content_tests_dir.glob('test_*.py'))
                metrics['content_api_test_files'] = len(content_test_files)
                details.append(f"✅ 統合コンテンツAPIテストファイル数: {len(content_test_files)}")
            else:
                details.append("❌ 統合コンテンツAPIテストが不在")
                status = 'FAIL'

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="REQ-5",
            title="包括的テスティング統合",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def verify_non_functional_requirements(self) -> VerificationResult:
        """非機能要件の検証"""
        start_time = time.time()
        details = []
        metrics = {}
        status = 'PASS'

        try:
            # 1. ドキュメント確認
            docs_to_check = [
                'backend/README.md',
                'docs/backend-architecture.md',
                'docs/developer-onboarding.md'
            ]

            docs_existing = 0
            for doc_path in docs_to_check:
                full_path = self.project_root / doc_path
                if full_path.exists():
                    docs_existing += 1
                    details.append(f"✅ ドキュメント存在: {doc_path}")
                else:
                    details.append(f"❌ ドキュメント不在: {doc_path}")
                    status = 'FAIL'

            metrics['documentation_files'] = docs_existing

            # 2. SQLマイグレーションファイル確認
            migrations_dir = self.project_root / 'supabase' / 'migrations'
            if migrations_dir.exists():
                migration_files = list(migrations_dir.glob('*.sql'))
                metrics['migration_files'] = len(migration_files)
                details.append(f"✅ SQLマイグレーションファイル数: {len(migration_files)}")
            else:
                details.append("❌ マイグレーションディレクトリが不在")
                status = 'FAIL'

            # 3. 監視システム確認
            monitoring_dir = self.project_root / 'backend' / 'monitoring'
            if monitoring_dir.exists():
                monitoring_files = list(monitoring_dir.glob('*.py'))
                metrics['monitoring_files'] = len(monitoring_files)
                details.append(f"✅ 監視システムファイル数: {len(monitoring_files)}")
            else:
                details.append("❌ 監視システムディレクトリが不在")
                status = 'FAIL'

        except Exception as e:
            details.append(f"❌ 検証エラー: {str(e)}")
            status = 'FAIL'

        duration = (time.time() - start_time) * 1000

        return VerificationResult(
            requirement_id="NFR",
            title="非機能要件",
            status=status,
            details=details,
            metrics=metrics,
            duration_ms=duration
        )

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """包括的検証の実行"""
        self.logger.info("🔍 包括的要件検証を開始します...")

        # 全検証を並列実行
        verification_tasks = [
            self.verify_requirement_1_edge_functions(),
            self.verify_requirement_2_ml_pipeline(),
            self.verify_requirement_3_data_pipeline(),
            self.verify_requirement_4_directory_structure(),
            self.verify_requirement_5_testing(),
            self.verify_non_functional_requirements()
        ]

        self.results = await asyncio.gather(*verification_tasks)

        # 結果集計
        total_requirements = len(self.results)
        passed_requirements = len([r for r in self.results if r.status == 'PASS'])
        failed_requirements = len([r for r in self.results if r.status == 'FAIL'])
        warning_requirements = len([r for r in self.results if r.status == 'WARNING'])

        overall_status = 'PASS'
        if failed_requirements > 0:
            overall_status = 'FAIL'
        elif warning_requirements > 0:
            overall_status = 'WARNING'

        summary = {
            'overall_status': overall_status,
            'total_requirements': total_requirements,
            'passed_requirements': passed_requirements,
            'failed_requirements': failed_requirements,
            'warning_requirements': warning_requirements,
            'success_rate': (passed_requirements / total_requirements) * 100,
            'total_duration_ms': sum(r.duration_ms for r in self.results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [asdict(result) for result in self.results]
        }

        return summary

    def generate_verification_report(self, summary: Dict[str, Any]) -> str:
        """検証レポートの生成"""
        overall_emoji = {
            'PASS': '✅',
            'WARNING': '⚠️',
            'FAIL': '❌'
        }[summary['overall_status']]

        report = f"""
# 🔍 Backend Refactoring 包括的要件検証レポート

## 📊 検証サマリー

**全体ステータス**: {overall_emoji} {summary['overall_status']}
**成功率**: {summary['success_rate']:.1f}% ({summary['passed_requirements']}/{summary['total_requirements']})
**検証時間**: {summary['total_duration_ms']:.0f}ms
**実行日時**: {summary['timestamp']}

### 要件別結果

| 要件ID | タイトル | ステータス | 実行時間 |
|--------|----------|-----------|----------|
"""

        for result in self.results:
            status_emoji = {
                'PASS': '✅',
                'WARNING': '⚠️',
                'FAIL': '❌'
            }[result.status]

            report += f"| {result.requirement_id} | {result.title} | {status_emoji} {result.status} | {result.duration_ms:.0f}ms |\n"

        report += "\n## 📋 詳細結果\n\n"

        for result in self.results:
            report += f"### {result.requirement_id}: {result.title}\n\n"
            report += f"**ステータス**: {result.status}\n\n"

            if result.details:
                report += "**詳細**:\n"
                for detail in result.details:
                    report += f"- {detail}\n"
                report += "\n"

            if result.metrics:
                report += "**メトリクス**:\n"
                for key, value in result.metrics.items():
                    report += f"- {key}: {value}\n"
                report += "\n"

        report += f"""
## 🎯 検証結論

リファクタリングプロジェクトの要件検証が完了しました。

### 🏆 成果
- **実装要件**: {summary['passed_requirements']}/{summary['total_requirements']} 達成
- **成功率**: {summary['success_rate']:.1f}%
- **プロジェクトステータス**: {summary['overall_status']}

### 📈 次のステップ
{"✅ 全要件達成により本番デプロイメント準備完了" if summary['overall_status'] == 'PASS' else "⚠️  警告またはエラーの解決が必要"}

---
**Generated by**: Comprehensive Verification System
**Version**: 1.0
**Report Type**: Backend Refactoring Requirements Verification
"""

        return report

async def main():
    """メイン実行関数"""
    verifier = ComprehensiveVerifier()

    try:
        # 包括的検証実行
        summary = await verifier.run_comprehensive_verification()

        # レポート生成
        report = verifier.generate_verification_report(summary)

        # レポート出力
        print(report)

        # レポートファイル保存
        report_dir = verifier.project_root / 'backend' / 'reports'
        report_dir.mkdir(exist_ok=True)

        report_file = report_dir / 'comprehensive_verification_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 詳細レポートを保存しました: {report_file}")

        # 結果に基づく終了コード
        exit_code = 0 if summary['overall_status'] == 'PASS' else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ 検証中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
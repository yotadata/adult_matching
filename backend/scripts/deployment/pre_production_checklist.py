#!/usr/bin/env python3
"""
本番デプロイメント前チェックリスト

リファクタリングされたシステムの本番環境デプロイメント準備を確認します。
"""

import asyncio
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class CheckResult:
    check_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING', 'SKIP'
    details: List[str]
    critical: bool = False
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class ProductionReadinessChecker:
    """本番環境準備チェッカー"""

    def __init__(self):
        self.project_root = project_root
        self.results: List[CheckResult] = []
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('production_checker')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def check_supabase_connection(self) -> CheckResult:
        """Supabase接続確認"""
        details = []
        status = 'PASS'

        try:
            # Supabase CLIの存在確認
            result = subprocess.run(['supabase', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                details.append(f"✅ Supabase CLI利用可能: {result.stdout.strip()}")
            else:
                details.append("❌ Supabase CLIが利用できません")
                status = 'FAIL'

            # Supabase ステータス確認
            try:
                status_result = subprocess.run(['supabase', 'status'],
                                             capture_output=True, text=True, timeout=30)

                if status_result.returncode == 0:
                    details.append("✅ Supabaseローカル環境が動作中")
                    # サービス状態を詳細確認
                    if 'Database URL' in status_result.stdout:
                        details.append("✅ データベース接続確認済み")
                    if 'API URL' in status_result.stdout:
                        details.append("✅ API URL設定確認済み")
                else:
                    details.append("⚠️  Supabaseローカル環境が停止中")
                    details.append("推奨: supabase start で起動してください")
                    if status == 'PASS':
                        status = 'WARNING'

            except subprocess.TimeoutExpired:
                details.append("⚠️  Supabaseステータス確認がタイムアウト")
                if status == 'PASS':
                    status = 'WARNING'

        except FileNotFoundError:
            details.append("❌ Supabase CLIがインストールされていません")
            status = 'FAIL'
        except Exception as e:
            details.append(f"❌ Supabase確認中にエラー: {str(e)}")
            status = 'FAIL'

        recommendations = []
        if status == 'FAIL':
            recommendations.append("Supabase CLIをインストールしてください")
        elif status == 'WARNING':
            recommendations.append("本番デプロイ前にSupabaseサービスを起動してください")

        return CheckResult(
            check_name="Supabase接続確認",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def check_edge_functions_deployment(self) -> CheckResult:
        """Edge Functions デプロイメント準備確認"""
        details = []
        status = 'PASS'

        functions_dir = self.project_root / 'supabase' / 'functions'

        # 必須Edge Functions確認
        required_functions = [
            'recommendations/enhanced_two_tower',
            'user-management/likes',
            'user-management/embeddings',
            'user-management/account',
            'content/feed'
        ]

        missing_functions = []
        for func_path in required_functions:
            func_dir = functions_dir / func_path
            index_file = func_dir / 'index.ts'

            if func_dir.exists() and index_file.exists():
                details.append(f"✅ Edge Function準備完了: {func_path}")
            else:
                details.append(f"❌ Edge Function不在: {func_path}")
                missing_functions.append(func_path)
                status = 'FAIL'

        # 共有ユーティリティ確認
        shared_dir = functions_dir / '_shared'
        shared_modules = ['auth.ts', 'database.ts', 'validation.ts', 'monitoring.ts']

        for module in shared_modules:
            if (shared_dir / module).exists():
                details.append(f"✅ 共有モジュール準備完了: {module}")
            else:
                details.append(f"⚠️  共有モジュール不在: {module}")
                if status == 'PASS':
                    status = 'WARNING'

        # TypeScript構文チェック（可能な場合）
        try:
            # deno check コマンドが利用可能かチェック
            ts_files = list(functions_dir.rglob('*.ts'))
            details.append(f"📊 TypeScriptファイル総数: {len(ts_files)}")

        except Exception as e:
            details.append(f"⚠️  TypeScript構文チェックをスキップ: {str(e)}")

        recommendations = []
        if missing_functions:
            recommendations.extend([
                f"不在のEdge Function実装が必要: {', '.join(missing_functions)}",
                "backend/scripts/deployment/deploy_edge_functions.sh を確認してください"
            ])

        return CheckResult(
            check_name="Edge Functions デプロイメント準備",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def check_database_migrations(self) -> CheckResult:
        """データベースマイグレーション確認"""
        details = []
        status = 'PASS'

        migrations_dir = self.project_root / 'supabase' / 'migrations'

        if not migrations_dir.exists():
            details.append("❌ マイグレーションディレクトリが存在しません")
            return CheckResult(
                check_name="データベースマイグレーション確認",
                status='FAIL',
                details=details,
                critical=True,
                recommendations=["supabase/migrations/ ディレクトリを作成してください"]
            )

        # マイグレーションファイル確認
        migration_files = sorted(list(migrations_dir.glob('*.sql')))

        if len(migration_files) == 0:
            details.append("⚠️  マイグレーションファイルが見つかりません")
            status = 'WARNING'
        else:
            details.append(f"✅ マイグレーションファイル数: {len(migration_files)}")

            # 重要なマイグレーション確認
            important_migrations = [
                'personalized_feed_functions',
                'missing_rpc_functions',
                'schema_dependencies'
            ]

            found_migrations = []
            for migration_file in migration_files:
                file_content = migration_file.read_text(encoding='utf-8')
                for important in important_migrations:
                    if important in migration_file.name or important in file_content:
                        found_migrations.append(important)
                        break

            for important in important_migrations:
                if important in found_migrations:
                    details.append(f"✅ 重要マイグレーション確認: {important}")
                else:
                    details.append(f"⚠️  重要マイグレーション不在: {important}")
                    if status == 'PASS':
                        status = 'WARNING'

        # マイグレーション実行状態確認
        try:
            # supabase db diff コマンドで未適用マイグレーション確認
            result = subprocess.run(['supabase', 'db', 'diff'],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                if result.stdout.strip():
                    details.append("⚠️  未適用のデータベース変更があります")
                    if status == 'PASS':
                        status = 'WARNING'
                else:
                    details.append("✅ データベーススキーマは最新状態です")
            else:
                details.append("⚠️  データベース差分確認ができませんでした")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            details.append("⚠️  データベース差分確認をスキップ")

        recommendations = []
        if status != 'PASS':
            recommendations.extend([
                "本番デプロイ前にマイグレーションを実行してください",
                "supabase db push コマンドで適用可能です"
            ])

        return CheckResult(
            check_name="データベースマイグレーション確認",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def check_environment_configuration(self) -> CheckResult:
        """環境設定確認"""
        details = []
        status = 'PASS'

        # 必要な環境変数確認
        required_env_vars = [
            'SUPABASE_URL',
            'SUPABASE_ANON_KEY',
            'SUPABASE_SERVICE_ROLE_KEY'
        ]

        missing_env_vars = []
        for env_var in required_env_vars:
            if os.getenv(env_var):
                details.append(f"✅ 環境変数設定済み: {env_var}")
            else:
                details.append(f"❌ 環境変数未設定: {env_var}")
                missing_env_vars.append(env_var)
                status = 'FAIL'

        # 設定ファイル確認
        config_files = [
            'supabase/config.toml',
            '.env.example'
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                details.append(f"✅ 設定ファイル存在: {config_file}")
            else:
                details.append(f"⚠️  設定ファイル不在: {config_file}")
                if status == 'PASS':
                    status = 'WARNING'

        # DMM API設定確認（オプション）
        dmm_api_vars = ['DMM_API_ID', 'DMM_AFFILIATE_ID']
        dmm_configured = all(os.getenv(var) for var in dmm_api_vars)

        if dmm_configured:
            details.append("✅ DMM API設定済み")
        else:
            details.append("⚠️  DMM API設定未完了（外部データ同期に影響）")

        recommendations = []
        if missing_env_vars:
            recommendations.extend([
                f"必須環境変数を設定してください: {', '.join(missing_env_vars)}",
                ".env ファイルを作成し、必要な値を設定してください"
            ])
        if not dmm_configured:
            recommendations.append("DMM API連携が必要な場合は、API キーを設定してください")

        return CheckResult(
            check_name="環境設定確認",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def check_monitoring_system(self) -> CheckResult:
        """監視システム準備確認"""
        details = []
        status = 'PASS'

        monitoring_dir = self.project_root / 'backend' / 'monitoring'

        if not monitoring_dir.exists():
            details.append("❌ 監視システムディレクトリが存在しません")
            return CheckResult(
                check_name="監視システム準備確認",
                status='FAIL',
                details=details,
                critical=False,
                recommendations=["監視システムを実装してください"]
            )

        # 監視モジュール確認
        monitoring_modules = [
            'system_monitor.py',
            'integration_monitor.py'
        ]

        for module in monitoring_modules:
            module_path = monitoring_dir / module
            if module_path.exists():
                details.append(f"✅ 監視モジュール存在: {module}")
            else:
                details.append(f"⚠️  監視モジュール不在: {module}")
                if status == 'PASS':
                    status = 'WARNING'

        # Python依存関係確認
        try:
            import prometheus_client
            details.append("✅ Prometheus client利用可能")
        except ImportError:
            details.append("⚠️  Prometheus client未インストール")
            if status == 'PASS':
                status = 'WARNING'

        # 監視設定確認
        prometheus_port = os.getenv('PROMETHEUS_PORT', '8000')
        details.append(f"📊 Prometheusポート設定: {prometheus_port}")

        recommendations = []
        if status != 'PASS':
            recommendations.extend([
                "監視システムの依存関係をインストールしてください",
                "pip install prometheus_client を実行してください"
            ])

        return CheckResult(
            check_name="監視システム準備確認",
            status=status,
            details=details,
            critical=False,
            recommendations=recommendations
        )

    async def check_test_coverage(self) -> CheckResult:
        """テストカバレッジ確認"""
        details = []
        status = 'PASS'

        tests_dir = self.project_root / 'backend' / 'tests'

        if not tests_dir.exists():
            details.append("❌ テストディレクトリが存在しません")
            return CheckResult(
                check_name="テストカバレッジ確認",
                status='FAIL',
                details=details,
                critical=True,
                recommendations=["テストスイートを実装してください"]
            )

        # テストファイル数確認
        test_categories = ['unit', 'integration', 'e2e']
        total_tests = 0

        for category in test_categories:
            category_dir = tests_dir / category
            if category_dir.exists():
                test_files = list(category_dir.rglob('test_*.py'))
                total_tests += len(test_files)
                details.append(f"✅ {category}テスト: {len(test_files)}ファイル")
            else:
                details.append(f"⚠️  {category}テストディレクトリ不在")
                if status == 'PASS':
                    status = 'WARNING'

        details.append(f"📊 総テストファイル数: {total_tests}")

        # 統合コンテンツAPIテスト特別確認
        content_tests = tests_dir / 'integration' / 'content'
        if content_tests.exists():
            content_test_files = list(content_tests.glob('test_*.py'))
            details.append(f"✅ 統合コンテンツAPIテスト: {len(content_test_files)}ファイル")
        else:
            details.append("❌ 統合コンテンツAPIテストが不在")
            status = 'FAIL'

        # pytest実行可能性確認
        try:
            result = subprocess.run(['python', '-m', 'pytest', '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details.append("✅ pytest実行可能")
            else:
                details.append("⚠️  pytest実行に問題があります")
                if status == 'PASS':
                    status = 'WARNING'
        except Exception:
            details.append("⚠️  pytest確認をスキップ")

        recommendations = []
        if status == 'FAIL':
            recommendations.extend([
                "不足しているテストスイートを実装してください",
                "本番デプロイ前に全テストの実行を推奨します"
            ])
        elif status == 'WARNING':
            recommendations.append("pytest backend/tests/ で全テストを実行してください")

        return CheckResult(
            check_name="テストカバレッジ確認",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def check_security_requirements(self) -> CheckResult:
        """セキュリティ要件確認"""
        details = []
        status = 'PASS'

        # 認証システム確認
        auth_files = [
            'supabase/functions/_shared/auth.ts',
            'supabase/functions/_shared/validation.ts'
        ]

        for auth_file in auth_files:
            auth_path = self.project_root / auth_file
            if auth_path.exists():
                details.append(f"✅ 認証モジュール存在: {auth_file}")
            else:
                details.append(f"❌ 認証モジュール不在: {auth_file}")
                status = 'FAIL'

        # RLSポリシー確認（マイグレーションファイルから）
        migrations_dir = self.project_root / 'supabase' / 'migrations'
        rls_policies_found = False

        if migrations_dir.exists():
            for migration_file in migrations_dir.glob('*.sql'):
                content = migration_file.read_text(encoding='utf-8')
                if 'CREATE POLICY' in content or 'ALTER TABLE' in content and 'RLS' in content:
                    rls_policies_found = True
                    break

            if rls_policies_found:
                details.append("✅ RLSポリシー設定確認済み")
            else:
                details.append("⚠️  RLSポリシー設定が見つかりません")
                if status == 'PASS':
                    status = 'WARNING'

        # 環境変数セキュリティ
        sensitive_vars = ['SUPABASE_SERVICE_ROLE_KEY', 'DMM_API_ID']
        for var in sensitive_vars:
            if os.getenv(var):
                # 値の一部のみ表示（セキュリティのため）
                value = os.getenv(var)
                masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                details.append(f"✅ 機密環境変数設定済み: {var}={masked_value}")
            else:
                details.append(f"⚠️  機密環境変数未設定: {var}")

        recommendations = []
        if status == 'FAIL':
            recommendations.extend([
                "認証・認可システムを完全に実装してください",
                "セキュリティチェックリストを確認してください"
            ])
        elif status == 'WARNING':
            recommendations.extend([
                "RLSポリシーが適切に設定されているか確認してください",
                "機密環境変数を適切に設定してください"
            ])

        return CheckResult(
            check_name="セキュリティ要件確認",
            status=status,
            details=details,
            critical=True,
            recommendations=recommendations
        )

    async def run_production_readiness_check(self) -> Dict[str, Any]:
        """本番環境準備チェック実行"""
        self.logger.info("🚀 本番デプロイメント準備チェックを開始します...")

        # 全チェックを並列実行
        check_tasks = [
            self.check_supabase_connection(),
            self.check_edge_functions_deployment(),
            self.check_database_migrations(),
            self.check_environment_configuration(),
            self.check_monitoring_system(),
            self.check_test_coverage(),
            self.check_security_requirements()
        ]

        self.results = await asyncio.gather(*check_tasks)

        # 結果集計
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == 'PASS'])
        failed_checks = len([r for r in self.results if r.status == 'FAIL'])
        warning_checks = len([r for r in self.results if r.status == 'WARNING'])
        critical_failures = len([r for r in self.results if r.status == 'FAIL' and r.critical])

        overall_status = 'READY'
        if critical_failures > 0:
            overall_status = 'NOT_READY'
        elif failed_checks > 0 or warning_checks > 2:
            overall_status = 'REVIEW_NEEDED'

        summary = {
            'overall_status': overall_status,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'warning_checks': warning_checks,
            'critical_failures': critical_failures,
            'readiness_score': (passed_checks / total_checks) * 100,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [asdict(result) for result in self.results]
        }

        return summary

    def generate_readiness_report(self, summary: Dict[str, Any]) -> str:
        """本番準備レポート生成"""
        status_emoji = {
            'READY': '✅',
            'REVIEW_NEEDED': '⚠️',
            'NOT_READY': '❌'
        }[summary['overall_status']]

        report = f"""
# 🚀 本番デプロイメント準備チェックレポート

## 📊 準備状況サマリー

**準備ステータス**: {status_emoji} {summary['overall_status']}
**準備スコア**: {summary['readiness_score']:.1f}% ({summary['passed_checks']}/{summary['total_checks']})
**重要な問題**: {summary['critical_failures']}件
**チェック時間**: {summary['timestamp']}

### チェック結果概要

| チェック項目 | ステータス | 重要度 |
|-------------|-----------|--------|
"""

        for result in self.results:
            status_emoji = {
                'PASS': '✅',
                'WARNING': '⚠️',
                'FAIL': '❌'
            }[result.status]

            critical_mark = "🔴 重要" if result.critical else "⚪ 通常"

            report += f"| {result.check_name} | {status_emoji} {result.status} | {critical_mark} |\n"

        report += f"\n## 📋 詳細チェック結果\n\n"

        for result in self.results:
            report += f"### {result.check_name}\n\n"
            report += f"**ステータス**: {result.status}\n"
            report += f"**重要度**: {'🔴 クリティカル' if result.critical else '⚪ 通常'}\n\n"

            if result.details:
                report += "**詳細**:\n"
                for detail in result.details:
                    report += f"- {detail}\n"
                report += "\n"

            if result.recommendations:
                report += "**推奨事項**:\n"
                for rec in result.recommendations:
                    report += f"- {rec}\n"
                report += "\n"

        # 最終判定
        if summary['overall_status'] == 'READY':
            report += f"""
## ✅ 本番デプロイメント承認

**結論**: 本番環境へのデプロイメント準備が完了しています。

### 次のステップ
1. 最終デプロイメント実行
2. デプロイメント後監視開始
3. 本番稼働開始

**承認者**: Production Readiness Checker
**承認日時**: {summary['timestamp']}
"""

        elif summary['overall_status'] == 'REVIEW_NEEDED':
            report += f"""
## ⚠️ 本番デプロイメント要検討

**結論**: 一部の警告事項があります。検討後にデプロイメント可能です。

### 対応必要事項
"""
            for result in self.results:
                if result.status in ['WARNING', 'FAIL'] and result.recommendations:
                    report += f"\n**{result.check_name}**:\n"
                    for rec in result.recommendations:
                        report += f"- {rec}\n"

        else:
            report += f"""
## ❌ 本番デプロイメント不可

**結論**: 重要な問題があります。解決してから再度チェックしてください。

### 必須対応事項
"""
            for result in self.results:
                if result.status == 'FAIL' and result.critical:
                    report += f"\n**{result.check_name}**:\n"
                    for rec in result.recommendations:
                        report += f"- {rec}\n"

        report += f"""

---
**Generated by**: Production Readiness Checker v1.0
**Report Type**: Backend Refactoring Production Deployment Check
**Generated at**: {summary['timestamp']}
"""

        return report

async def main():
    """メイン実行関数"""
    checker = ProductionReadinessChecker()

    try:
        # 本番準備チェック実行
        summary = await checker.run_production_readiness_check()

        # レポート生成
        report = checker.generate_readiness_report(summary)

        # レポート出力
        print(report)

        # レポートファイル保存
        report_dir = checker.project_root / 'backend' / 'reports'
        report_dir.mkdir(exist_ok=True)

        report_file = report_dir / 'production_readiness_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 詳細レポートを保存しました: {report_file}")

        # 結果に基づく終了コード
        exit_code = 0 if summary['overall_status'] == 'READY' else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ 本番準備チェック中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
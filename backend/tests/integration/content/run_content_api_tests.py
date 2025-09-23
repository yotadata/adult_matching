#!/usr/bin/env python3
"""
統合コンテンツAPIテストランナー

全コンテンツAPIテストの統合実行スクリプト
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ContentAPITestRunner:
    """コンテンツAPIテストランナー"""

    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.project_root = project_root
        self.results = {}

    def run_test_suite(
        self,
        test_file: str,
        test_name: str,
        markers: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """テストスイートの実行"""
        print(f"\n🚀 Running {test_name}...")
        print(f"   File: {test_file}")

        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / test_file),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            f"--junitxml={self.test_dir}/reports/{test_name}_results.xml",
            f"--html={self.test_dir}/reports/{test_name}_report.html",
            "--self-contained-html"
        ]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30分タイムアウト
            )

            end_time = time.time()
            duration = end_time - start_time

            test_result = {
                'name': test_name,
                'file': test_file,
                'success': result.returncode == 0,
                'duration_seconds': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if test_result['success']:
                print(f"✅ {test_name} completed successfully in {duration:.2f}s")
            else:
                print(f"❌ {test_name} failed after {duration:.2f}s")
                print(f"   Return code: {result.returncode}")

            return test_result

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} timed out after 30 minutes")
            return {
                'name': test_name,
                'file': test_file,
                'success': False,
                'duration_seconds': 1800,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out'
            }

        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            return {
                'name': test_name,
                'file': test_file,
                'success': False,
                'duration_seconds': 0,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }

    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        print("🔧 Setting up test environment...")

        # レポートディレクトリ作成
        reports_dir = self.test_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # 環境変数設定
        os.environ['CONTENT_API_TEST_MODE'] = 'true'
        os.environ['PYTHONPATH'] = str(self.project_root)

        # 必要なパッケージの確認
        required_packages = [
            'pytest', 'pytest-asyncio', 'httpx', 'psutil'
        ]

        print("   Checking required packages...")
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"   ⚠️  Missing packages: {', '.join(missing_packages)}")
            print(f"   Install with: pip install {' '.join(missing_packages)}")
            return False

        print("   ✅ Test environment ready")
        return True

    def run_comprehensive_tests(self, include_slow: bool = False) -> Dict[str, Any]:
        """包括的テストの実行"""
        if not self.setup_test_environment():
            return {'success': False, 'error': 'Failed to setup test environment'}

        print("\n" + "="*60)
        print("🧪 CONTENT API COMPREHENSIVE TEST SUITE")
        print("="*60)

        test_suites = [
            {
                'file': 'test_content_api_comprehensive.py',
                'name': 'comprehensive',
                'description': 'Core functionality tests',
                'markers': ['not slow'] if not include_slow else None
            },
            {
                'file': 'test_content_api_edge_cases.py',
                'name': 'edge_cases',
                'description': 'Edge cases and error handling',
                'markers': ['not slow'] if not include_slow else None
            },
            {
                'file': 'test_content_api_performance.py',
                'name': 'performance',
                'description': 'Performance and load tests',
                'markers': ['not slow'] if not include_slow else None
            }
        ]

        results = []
        total_start_time = time.time()

        for suite in test_suites:
            print(f"\n📋 {suite['description']}")
            result = self.run_test_suite(
                test_file=suite['file'],
                test_name=suite['name'],
                markers=suite['markers']
            )
            results.append(result)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # 結果サマリー
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]

        summary = {
            'success': len(failed_tests) == 0,
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'total_duration_seconds': total_duration,
            'results': results
        }

        self.print_summary(summary)
        self.save_results(summary)

        return summary

    def run_specific_test(self, test_category: str, include_slow: bool = False) -> Dict[str, Any]:
        """特定カテゴリのテスト実行"""
        if not self.setup_test_environment():
            return {'success': False, 'error': 'Failed to setup test environment'}

        test_files = {
            'comprehensive': 'test_content_api_comprehensive.py',
            'edge_cases': 'test_content_api_edge_cases.py',
            'performance': 'test_content_api_performance.py'
        }

        if test_category not in test_files:
            return {
                'success': False,
                'error': f'Unknown test category: {test_category}. Available: {list(test_files.keys())}'
            }

        print(f"\n🎯 Running {test_category} tests...")

        markers = ['not slow'] if not include_slow else None
        result = self.run_test_suite(
            test_file=test_files[test_category],
            test_name=test_category,
            markers=markers
        )

        summary = {
            'success': result['success'],
            'total_tests': 1,
            'successful_tests': 1 if result['success'] else 0,
            'failed_tests': 0 if result['success'] else 1,
            'total_duration_seconds': result['duration_seconds'],
            'results': [result]
        }

        self.print_summary(summary)
        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """結果サマリーの表示"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)

        status_emoji = "✅" if summary['success'] else "❌"
        print(f"{status_emoji} Overall Status: {'PASSED' if summary['success'] else 'FAILED'}")
        print(f"📈 Tests: {summary['successful_tests']}/{summary['total_tests']} passed")
        print(f"⏱️  Total Duration: {summary['total_duration_seconds']:.2f} seconds")

        if summary['failed_tests'] > 0:
            print(f"\n❌ Failed Tests ({summary['failed_tests']}):")
            for result in summary['results']:
                if not result['success']:
                    print(f"   • {result['name']} (exit code: {result['returncode']})")

        print(f"\n📄 Detailed reports available in: {self.test_dir}/reports/")

    def save_results(self, summary: Dict[str, Any]):
        """結果の保存"""
        results_file = self.test_dir / "reports" / "test_summary.json"

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"💾 Results saved to: {results_file}")

    def run_quick_smoke_test(self) -> bool:
        """クイックスモークテスト"""
        print("\n🚀 Running quick smoke test...")

        result = self.run_test_suite(
            test_file='test_content_api_comprehensive.py',
            test_name='smoke_test',
            markers=['not slow', 'not performance'],
            verbose=False
        )

        return result['success']


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Content API Test Runner')

    parser.add_argument(
        'command',
        choices=['all', 'comprehensive', 'edge_cases', 'performance', 'smoke'],
        help='Test command to run'
    )

    parser.add_argument(
        '--include-slow',
        action='store_true',
        help='Include slow running tests'
    )

    parser.add_argument(
        '--test-dir',
        type=Path,
        help='Test directory path (default: current directory)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    runner = ContentAPITestRunner(test_dir=args.test_dir)

    if args.command == 'smoke':
        success = runner.run_quick_smoke_test()
        sys.exit(0 if success else 1)

    elif args.command == 'all':
        summary = runner.run_comprehensive_tests(include_slow=args.include_slow)

    else:
        summary = runner.run_specific_test(args.command, include_slow=args.include_slow)

    sys.exit(0 if summary['success'] else 1)


if __name__ == "__main__":
    main()
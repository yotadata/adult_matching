#!/usr/bin/env python3
"""
Comprehensive Test Runner

統合テストフレームワーク実行スクリプト
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

# バックエンドルートをパスに追加
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))


class TestRunner:
    """包括的テストランナー"""
    
    def __init__(self):
        self.backend_root = backend_root
        self.test_dir = backend_root / "tests"
        self.coverage_dir = self.test_dir / "coverage_html"
        self.results = {}
    
    def run_unit_tests(self, verbose: bool = False) -> dict:
        """ユニットテスト実行"""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit"),
            "-m", "unit",
            "--tb=short",
            "--cov=backend",
            f"--cov-report=html:{self.coverage_dir}",
            "--cov-report=term-missing",
            "--durations=10",
            "--junitxml=test_results_unit.xml"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.backend_root, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "name": "Unit Tests",
            "success": result.returncode == 0,
            "duration": duration,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }
    
    def run_integration_tests(self, verbose: bool = False) -> dict:
        """統合テスト実行"""
        print("🔗 Running Integration Tests...")
        
        # 統合テスト用環境変数設定
        env = os.environ.copy()
        env["RUN_INTEGRATION_TESTS"] = "1"
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "--tb=short",
            "--durations=10",
            "--junitxml=test_results_integration.xml"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.backend_root, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        
        return {
            "name": "Integration Tests",
            "success": result.returncode == 0,
            "duration": duration,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }
    
    def run_performance_tests(self, verbose: bool = False) -> dict:
        """パフォーマンステスト実行"""
        print("⚡ Running Performance Tests...")
        
        env = os.environ.copy()
        env["RUN_PERFORMANCE_TESTS"] = "1"
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "performance",
            "--tb=short",
            "--durations=0",
            "--junitxml=test_results_performance.xml"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.backend_root, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        
        return {
            "name": "Performance Tests",
            "success": result.returncode == 0,
            "duration": duration,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }
    
    def run_smoke_tests(self, verbose: bool = False) -> dict:
        """スモークテスト実行"""
        print("💨 Running Smoke Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "smoke",
            "--tb=line",
            "--durations=5",
            "--junitxml=test_results_smoke.xml"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.backend_root, capture_output=True, text=True)
        duration = time.time() - start_time
        
        return {
            "name": "Smoke Tests",
            "success": result.returncode == 0,
            "duration": duration,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }
    
    def run_all_tests(self, include_slow: bool = False, verbose: bool = False) -> dict:
        """全テスト実行"""
        print("🚀 Running All Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--tb=short",
            "--cov=backend",
            f"--cov-report=html:{self.coverage_dir}",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
            "--durations=10",
            "--junitxml=test_results_all.xml"
        ]
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        if verbose:
            cmd.append("-v")
        
        # 環境変数設定
        env = os.environ.copy()
        env["RUN_INTEGRATION_TESTS"] = "1"
        env["RUN_PERFORMANCE_TESTS"] = "1"
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.backend_root, capture_output=True, text=True, env=env)
        duration = time.time() - start_time
        
        return {
            "name": "All Tests",
            "success": result.returncode == 0,
            "duration": duration,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }
    
    def generate_coverage_report(self) -> dict:
        """カバレッジレポート生成"""
        print("📊 Generating Coverage Report...")
        
        # HTMLレポート確認
        index_file = self.coverage_dir / "index.html"
        
        if index_file.exists():
            print(f"📋 Coverage report available at: {index_file}")
            return {
                "success": True,
                "report_path": str(index_file),
                "message": "Coverage report generated successfully"
            }
        else:
            return {
                "success": False,
                "message": "Coverage report not found. Run tests with coverage first."
            }
    
    def create_test_summary(self) -> str:
        """テスト結果サマリー作成"""
        summary = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "backend_root": str(self.backend_root),
                "python_version": sys.version
            },
            "results": self.results,
            "summary": {
                "total_suites": len(self.results),
                "passed_suites": sum(1 for r in self.results.values() if r["success"]),
                "total_duration": sum(r["duration"] for r in self.results.values()),
                "overall_success": all(r["success"] for r in self.results.values())
            }
        }
        
        return json.dumps(summary, indent=2)
    
    def print_results_summary(self):
        """結果サマリー表示"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        total_duration = 0
        all_passed = True
        
        for result in self.results.values():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = result["duration"]
            total_duration += duration
            
            print(f"{status} {result['name']:<20} ({duration:.2f}s)")
            
            if not result["success"]:
                all_passed = False
                print(f"   Error: {result['errors'][:100]}...")
        
        print("-" * 60)
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Overall Result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        
        # カバレッジレポートのパス表示
        if self.coverage_dir.exists():
            print(f"Coverage Report: {self.coverage_dir / 'index.html'}")
        
        print("="*60)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Backend Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests only
  python run_tests.py --integration             # Run integration tests only
  python run_tests.py --performance             # Run performance tests only
  python run_tests.py --smoke                   # Run smoke tests only
  python run_tests.py --all                     # Run all tests
  python run_tests.py --all --include-slow      # Run all tests including slow ones
  python run_tests.py --coverage-only           # Generate coverage report only
        """
    )
    
    # テストタイプ選択
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests")
    test_group.add_argument("--smoke", action="store_true", help="Run smoke tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--coverage-only", action="store_true", help="Generate coverage report only")
    
    # オプション
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.coverage_only:
            result = runner.generate_coverage_report()
            if result["success"]:
                print(f"✅ {result['message']}")
                print(f"📋 Report: {result['report_path']}")
            else:
                print(f"❌ {result['message']}")
            return 0 if result["success"] else 1
        
        # テスト実行
        if args.unit:
            runner.results["unit"] = runner.run_unit_tests(args.verbose)
        elif args.integration:
            runner.results["integration"] = runner.run_integration_tests(args.verbose)
        elif args.performance:
            runner.results["performance"] = runner.run_performance_tests(args.verbose)
        elif args.smoke:
            runner.results["smoke"] = runner.run_smoke_tests(args.verbose)
        elif args.all:
            runner.results["all"] = runner.run_all_tests(args.include_slow, args.verbose)
        
        # カバレッジレポート生成
        coverage_result = runner.generate_coverage_report()
        
        # 結果表示
        runner.print_results_summary()
        
        # 結果保存
        if args.save_results:
            results_file = runner.test_dir / "test_results.json"
            with open(results_file, 'w') as f:
                f.write(runner.create_test_summary())
            print(f"📄 Results saved to: {results_file}")
        
        # 終了コード決定
        all_passed = all(r["success"] for r in runner.results.values())
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
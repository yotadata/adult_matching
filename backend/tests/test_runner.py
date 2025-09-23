"""
Test Runner

包括的テストスイート実行管理システム
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pytest


class TestCategory(Enum):
    """テストカテゴリ"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    ML = "ml"
    ALL = "all"


class TestResult(Enum):
    """テスト結果"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestRun:
    """テスト実行情報"""
    category: TestCategory
    start_time: float
    end_time: Optional[float] = None
    result: Optional[TestResult] = None
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    duration: float = 0.0
    coverage: Optional[float] = None
    exit_code: int = 0
    output: str = ""
    error: str = ""


class TestRunner:
    """包括的テストランナー"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.project_root = self.base_path.parent
        self.test_results: List[TestRun] = []
        
        # Test directories
        self.test_dirs = {
            TestCategory.UNIT: self.base_path / "unit",
            TestCategory.INTEGRATION: self.base_path / "integration", 
            TestCategory.E2E: self.base_path / "e2e",
            TestCategory.PERFORMANCE: self.base_path / "performance",
            TestCategory.ML: self.base_path / "ml"
        }
    
    def run_tests(self, categories: List[TestCategory], 
                  coverage: bool = True,
                  parallel: bool = False,
                  verbose: bool = False,
                  fail_fast: bool = False) -> List[TestRun]:
        """テスト実行"""
        print("🚀 包括的テストスイート実行開始")
        print("=" * 60)
        
        results = []
        
        for category in categories:
            if category == TestCategory.ALL:
                # Run all categories
                all_categories = [c for c in TestCategory if c != TestCategory.ALL]
                for cat in all_categories:
                    result = self._run_category(cat, coverage, parallel, verbose, fail_fast)
                    results.append(result)
                    if fail_fast and result.result == TestResult.FAILED:
                        break
            else:
                result = self._run_category(category, coverage, parallel, verbose, fail_fast)
                results.append(result)
                if fail_fast and result.result == TestResult.FAILED:
                    break
        
        self.test_results.extend(results)
        self._print_summary(results)
        return results
    
    def _run_category(self, category: TestCategory, 
                     coverage: bool = True,
                     parallel: bool = False,
                     verbose: bool = False,
                     fail_fast: bool = False) -> TestRun:
        """カテゴリ別テスト実行"""
        test_run = TestRun(category=category, start_time=time.time())
        
        test_dir = self.test_dirs.get(category)
        if not test_dir or not test_dir.exists():
            print(f"⚠️  テストディレクトリが見つかりません: {test_dir}")
            test_run.end_time = time.time()
            test_run.result = TestResult.SKIPPED
            test_run.duration = test_run.end_time - test_run.start_time
            return test_run
        
        print(f"\n📁 {category.value.upper()} テスト実行中...")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", str(test_dir)]
        
        # Add options
        if verbose:
            cmd.append("-v")
        
        if fail_fast:
            cmd.append("-x")
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if coverage:
            cmd.extend([
                "--cov=backend",
                f"--cov-report=html:htmlcov_{category.value}",
                "--cov-report=term-missing"
            ])
        
        # Add markers based on category
        if category == TestCategory.UNIT:
            cmd.extend(["-m", "unit"])
        elif category == TestCategory.INTEGRATION:
            cmd.extend(["-m", "integration"])
        elif category == TestCategory.PERFORMANCE:
            cmd.extend(["-m", "performance"])
        elif category == TestCategory.ML:
            cmd.extend(["-m", "ml"])
        
        # Add JSON report
        json_report = self.base_path / f"report_{category.value}.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            
            # Special environment for different test categories
            if category == TestCategory.INTEGRATION:
                env["RUN_INTEGRATION_TESTS"] = "1"
            elif category == TestCategory.PERFORMANCE:
                env["RUN_PERFORMANCE_TESTS"] = "1"
            
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            test_run.exit_code = result.returncode
            test_run.output = result.stdout
            test_run.error = result.stderr
            
            # Parse JSON report if available
            if json_report.exists():
                self._parse_json_report(test_run, json_report)
            
            # Determine result
            if result.returncode == 0:
                test_run.result = TestResult.PASSED
                print(f"✅ {category.value.upper()} テスト成功")
            else:
                test_run.result = TestResult.FAILED
                print(f"❌ {category.value.upper()} テスト失敗")
                if verbose:
                    print(f"Error output:\n{result.stderr}")
        
        except subprocess.TimeoutExpired:
            test_run.result = TestResult.ERROR
            test_run.error = "Test execution timed out"
            print(f"⏱️  {category.value.upper()} テストタイムアウト")
        
        except Exception as e:
            test_run.result = TestResult.ERROR
            test_run.error = str(e)
            print(f"💥 {category.value.upper()} テスト実行エラー: {e}")
        
        test_run.end_time = time.time()
        test_run.duration = test_run.end_time - test_run.start_time
        
        return test_run
    
    def _parse_json_report(self, test_run: TestRun, json_file: Path):
        """JSON レポート解析"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            summary = report.get('summary', {})
            test_run.tests_passed = summary.get('passed', 0)
            test_run.tests_failed = summary.get('failed', 0)
            test_run.tests_skipped = summary.get('skipped', 0)
            test_run.tests_total = summary.get('total', 0)
            
            # Extract coverage if available
            if 'coverage' in report:
                test_run.coverage = report['coverage'].get('percent_covered')
        
        except Exception as e:
            print(f"⚠️  JSON レポート解析エラー: {e}")
    
    def _print_summary(self, results: List[TestRun]):
        """テスト結果サマリー表示"""
        print("\n" + "=" * 60)
        print("📊 テスト結果サマリー")
        print("=" * 60)
        
        total_duration = sum(r.duration for r in results)
        total_tests = sum(r.tests_total for r in results)
        total_passed = sum(r.tests_passed for r in results)
        total_failed = sum(r.tests_failed for r in results)
        total_skipped = sum(r.tests_skipped for r in results)
        
        print(f"📝 総実行時間: {total_duration:.1f}秒")
        print(f"🎯 総テスト数: {total_tests}")
        print(f"✅ 成功: {total_passed}")
        print(f"❌ 失敗: {total_failed}")
        print(f"⏭️  スキップ: {total_skipped}")
        print(f"📈 成功率: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        
        print(f"\n📋 カテゴリ別詳細:")
        for result in results:
            status_icon = {
                TestResult.PASSED: "✅",
                TestResult.FAILED: "❌", 
                TestResult.SKIPPED: "⏭️",
                TestResult.ERROR: "💥"
            }.get(result.result, "❓")
            
            coverage_info = f" (カバレッジ: {result.coverage:.1f}%)" if result.coverage else ""
            
            print(f"  {status_icon} {result.category.value.upper()}: "
                  f"{result.tests_passed}/{result.tests_total} "
                  f"({result.duration:.1f}秒){coverage_info}")
        
        # Overall result
        print(f"\n🏁 全体結果: ", end="")
        if all(r.result == TestResult.PASSED for r in results):
            print("✅ 全テスト成功")
        elif any(r.result == TestResult.FAILED for r in results):
            print("❌ 一部テスト失敗")
        else:
            print("⚠️  テスト完了（スキップあり）")
    
    def run_specific_test(self, test_path: str, verbose: bool = True) -> TestRun:
        """特定テスト実行"""
        test_run = TestRun(category=TestCategory.UNIT, start_time=time.time())
        
        cmd = ["python", "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            test_run.exit_code = result.returncode
            test_run.output = result.stdout
            test_run.error = result.stderr
            test_run.result = TestResult.PASSED if result.returncode == 0 else TestResult.FAILED
            
        except Exception as e:
            test_run.result = TestResult.ERROR
            test_run.error = str(e)
        
        test_run.end_time = time.time()
        test_run.duration = test_run.end_time - test_run.start_time
        
        return test_run
    
    def generate_html_report(self, output_path: Path):
        """HTML レポート生成"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Results Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .test-category { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .passed { border-left-color: #28a745; }
        .failed { border-left-color: #dc3545; }
        .skipped { border-left-color: #ffc107; }
        .error { border-left-color: #fd7e14; }
        .metrics { display: flex; gap: 20px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Backend Test Results</h1>
        <p>Generated at: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div class="metrics">
            <div class="metric">
                <h3>{total_tests}</h3>
                <p>Total Tests</p>
            </div>
            <div class="metric">
                <h3>{total_passed}</h3>
                <p>Passed</p>
            </div>
            <div class="metric">
                <h3>{total_failed}</h3>
                <p>Failed</p>
            </div>
            <div class="metric">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
        </div>
    </div>
    
    <div class="details">
        <h2>📋 Category Details</h2>
        {category_details}
    </div>
</body>
</html>
"""
        
        # Calculate totals
        total_tests = sum(r.tests_total for r in self.test_results)
        total_passed = sum(r.tests_passed for r in self.test_results)
        total_failed = sum(r.tests_failed for r in self.test_results)
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Generate category details
        category_html = ""
        for result in self.test_results:
            status_class = result.result.value if result.result else "unknown"
            coverage_info = f"Coverage: {result.coverage:.1f}%" if result.coverage else "Coverage: N/A"
            
            category_html += f"""
            <div class="test-category {status_class}">
                <h3>{result.category.value.upper()}</h3>
                <p><strong>Status:</strong> {result.result.value if result.result else "Unknown"}</p>
                <p><strong>Tests:</strong> {result.tests_passed}/{result.tests_total}</p>
                <p><strong>Duration:</strong> {result.duration:.1f}s</p>
                <p><strong>{coverage_info}</strong></p>
            </div>
            """
        
        # Fill template
        final_html = html_content.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            success_rate=success_rate,
            category_details=category_html
        )
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"📄 HTML レポート生成: {output_path}")
    
    def export_results_json(self, output_path: Path):
        """結果をJSON形式でエクスポート"""
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": sum(r.tests_total for r in self.test_results),
                "total_passed": sum(r.tests_passed for r in self.test_results),
                "total_failed": sum(r.tests_failed for r in self.test_results),
                "total_skipped": sum(r.tests_skipped for r in self.test_results),
                "total_duration": sum(r.duration for r in self.test_results)
            },
            "categories": []
        }
        
        for result in self.test_results:
            export_data["categories"].append({
                "category": result.category.value,
                "result": result.result.value if result.result else None,
                "tests_total": result.tests_total,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "tests_skipped": result.tests_skipped,
                "duration": result.duration,
                "coverage": result.coverage,
                "exit_code": result.exit_code
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 JSON レポート生成: {output_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="包括的テストランナー")
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=[cat.value for cat in TestCategory],
        default=["all"],
        help="実行するテストカテゴリ"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="カバレッジ測定を無効化"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="並列実行を有効化"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="最初の失敗で停止"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("test_results"),
        help="結果出力ディレクトリ"
    )
    
    parser.add_argument(
        "--test-file",
        help="特定のテストファイルを実行"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    runner = TestRunner()
    
    if args.test_file:
        # Run specific test
        result = runner.run_specific_test(args.test_file, args.verbose)
        print(f"Test result: {result.result.value if result.result else 'Unknown'}")
        return result.exit_code
    
    # Parse categories
    categories = [TestCategory(cat) for cat in args.categories]
    
    # Run tests
    results = runner.run_tests(
        categories=categories,
        coverage=not args.no_coverage,
        parallel=args.parallel,
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )
    
    # Generate reports
    runner.generate_html_report(args.output_dir / "test_report.html")
    runner.export_results_json(args.output_dir / "test_results.json")
    
    # Return appropriate exit code
    if any(r.result == TestResult.FAILED for r in results):
        return 1
    elif any(r.result == TestResult.ERROR for r in results):
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
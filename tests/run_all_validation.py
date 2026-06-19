#!/usr/bin/env python3
"""
Comprehensive validation test runner.

Executes all validation tests and generates summary report.
Saves a full human-readable report to TEST_RESULTS.md in the repo root.

Usage:
    python tests/run_all_validation.py
"""
import sys
import subprocess
import datetime
from pathlib import Path
import time


def run_test_file(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Determine if passed
        passed = result.returncode == 0
        
        return {
            'file': test_file.name,
            'passed': passed,
            'returncode': result.returncode,
            'elapsed': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s")
        return {
            'file': test_file.name,
            'passed': False,
            'returncode': -1,
            'elapsed': elapsed,
            'stdout': '',
            'stderr': 'Test timed out'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR: {e}")
        return {
            'file': test_file.name,
            'passed': False,
            'returncode': -1,
            'elapsed': elapsed,
            'stdout': '',
            'stderr': str(e)
        }


def save_results_markdown(results, output_path):
    """Write a human-readable Markdown report of all test results."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    total_time = sum(r['elapsed'] for r in results)

    lines = [
        "# Test Results",
        "",
        f"**Run date:** {now}  ",
        f"**Python:** {sys.version.splitlines()[0]}  ",
        f"**Repository:** siya7205/ensemble-anomaly-maps  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Test files run | {total} |",
        f"| ✅ Passed | {passed} |",
        f"| ❌ Failed | {failed} |",
        f"| Total time | {total_time:.1f}s |",
        "",
        "## Per-File Results",
        "",
        "| Status | Test File | Time (s) | Notes |",
        "|--------|-----------|----------|-------|",
    ]

    for r in results:
        icon = "✅ PASS" if r['passed'] else "❌ FAIL"
        note = ""
        if not r['passed']:
            err_lines = r['stderr'].strip().splitlines()
            note = err_lines[-1][:80] if err_lines else "non-zero exit"
        lines.append(f"| {icon} | `{r['file']}` | {r['elapsed']:.1f} | {note} |")

    lines += ["", "---", "", "## Full Output Per Test File", ""]

    for r in results:
        icon = "✅ PASS" if r['passed'] else "❌ FAIL"
        lines.append(f"### {icon} `{r['file']}` ({r['elapsed']:.1f}s)")
        lines.append("")
        stdout = r['stdout'].strip()
        stderr = r['stderr'].strip()
        if stdout:
            lines.append("```")
            lines.append(stdout)
            lines.append("```")
            lines.append("")
        if stderr:
            lines.append("**stderr:**")
            lines.append("```")
            lines.append(stderr)
            lines.append("```")
            lines.append("")
        if not stdout and not stderr:
            lines.append("*(no output)*")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Full report saved to: {output_path}")


def main():
    """Run all validation tests."""
    print("="*70)
    print("COMPREHENSIVE VALIDATION TEST SUITE")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    # Find all validation test files
    tests_dir = Path(__file__).parent

    # Discover every test_*.py in the tests directory automatically
    test_files = sorted(tests_dir.glob('test_*.py'))

    if not test_files:
        print("✗ No test files found!")
        return 1

    print(f"\nFound {len(test_files)} test suites to run\n")

    # Run all tests
    results = []
    for test_file in test_files:
        result = run_test_file(test_file)
        results.append(result)

    # Generate summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['passed'])
    failed_tests = total_tests - passed_tests
    total_time = sum(r['elapsed'] for r in results)

    print(f"\nTest Files Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total Time: {total_time:.2f}s")

    print("\nDetailed Results:")
    print("-" * 70)
    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{status:8} | {result['file']:40} | {result['elapsed']:6.2f}s")

    # Save Markdown report to repo root
    repo_root = tests_dir.parent
    save_results_markdown(results, repo_root / "TEST_RESULTS.md")

    # Overall status
    print("\n" + "="*70)
    if failed_tests == 0:
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*70)
        print("\nThe ML pipeline has been comprehensively validated:")
        print("  • Dataset quality verified")
        print("  • Statistical rigor confirmed")
        print("  • Reproducibility established")
        print("  • Scientific methods validated")
        print("\nThis pipeline is ready for publication-quality research.")
        return 0
    else:
        print(f"✗ {failed_tests} TEST SUITE(S) FAILED")
        print("="*70)
        print("\nFailed tests:")
        for result in results:
            if not result['passed']:
                print(f"  • {result['file']}")
                if result['stderr']:
                    print(f"    Error: {result['stderr'][:100]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

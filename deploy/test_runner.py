"""
Test execution module for running pytest on Databricks clusters.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from .command_executor import CommandExecutor, TestResults

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CompositeTestResults:
    """Combined results from multiple test phases."""
    phases: Dict[str, TestResults]
    overall_success: bool = False
    total_passed: int = 0
    total_failed: int = 0
    
    def __post_init__(self):
        """Calculate overall statistics."""
        self.total_passed = sum(phase.passed for phase in self.phases.values())
        self.total_failed = sum(phase.failed for phase in self.phases.values())
        self.overall_success = self.total_failed == 0 and self.total_passed > 0
    
    def add_phase(self, name: str, results: TestResults):
        """Add test results for a phase."""
        self.phases[name] = results
        self.__post_init__()  # Recalculate stats
    
    @property
    def passed(self) -> bool:
        """Check if all tests passed."""
        return self.overall_success
    
    def get_summary(self) -> str:
        """Get summary string of results."""
        if not self.phases:
            return "No tests run"
        
        summary_parts = []
        for phase_name, results in self.phases.items():
            status = "✅" if results.success else "❌"
            summary_parts.append(f"{status} {phase_name}: {results.passed}P/{results.failed}F")
        
        overall = "✅ ALL PASSED" if self.overall_success else "❌ SOME FAILED"
        return f"{overall} | " + " | ".join(summary_parts)


class TestRunner:
    """Execute tests on Databricks and provide structured results."""
    
    def __init__(self, executor: CommandExecutor):
        """Initialize with command executor."""
        self.executor = executor
    
    def run_all_tests(self, markers: Optional[str] = None, fail_fast: bool = True) -> CompositeTestResults:
        """
        Run all test phases with optional fail-fast behavior.
        
        Args:
            markers: Optional pytest markers to filter tests
            fail_fast: Stop on first failure if True
            
        Returns:
            CompositeTestResults with results from all phases
        """
        results = CompositeTestResults(phases={})
        
        # Phase 1: Unit tests
        logger.info("🧪 Running unit tests...")
        console.print(Panel("🧪 Running Unit Tests", style="bold blue"))
        
        unit_cmd = "tests/ -m unit" if not markers else f"tests/ -m 'unit and ({markers})'"
        console.print(f"[dim]Command: pytest {unit_cmd}")
        
        unit_results = self.run_pytest(unit_cmd)
        results.add_phase("unit", unit_results)
        
        # Show unit test results
        if unit_results.success:
            console.print(f"[green]✅ Unit tests passed: {unit_results.passed} tests")
        else:
            console.print(f"[red]❌ Unit tests failed: {unit_results.failed} failed, {unit_results.passed} passed")
            console.print(f"[dim]Duration: {unit_results.duration:.1f}s")
        
        if not unit_results.success and fail_fast:
            logger.error("Unit tests failed, stopping (fail_fast=True)")
            return results
        
        # Phase 2: Integration tests (if unit tests passed or fail_fast is False)
        if markers != "unit":  # Allow skipping integration tests
            logger.info("🔄 Running integration tests...")
            console.print(Panel("🔄 Running Integration Tests", style="bold yellow"))
            
            integration_cmd = "tests/ -m integration" if not markers else f"tests/ -m 'integration and ({markers})'"
            console.print(f"[dim]Command: pytest {integration_cmd}")
            
            integration_results = self.run_pytest(integration_cmd)
            results.add_phase("integration", integration_results)
            
            # Show integration test results
            if integration_results.success:
                console.print(f"[green]✅ Integration tests passed: {integration_results.passed} tests")
            else:
                console.print(f"[red]❌ Integration tests failed: {integration_results.failed} failed, {integration_results.passed} passed")
                console.print(f"[dim]Duration: {integration_results.duration:.1f}s")
            
            if not integration_results.success and fail_fast:
                logger.error("Integration tests failed, stopping (fail_fast=True)")
                return results
        
        return results
    
    def run_pytest(self, test_spec: str) -> TestResults:
        """
        Execute pytest with the given specification.
        
        Args:
            test_spec: Test specification (e.g., "tests/", "tests/test_file.py -m unit")
            
        Returns:
            TestResults with parsed output
        """
        logger.info(f"Running pytest: {test_spec}")
        
        try:
            # Use the command executor's pytest runner
            return self.executor.run_pytest(test_spec)
            
        except Exception as e:
            logger.error(f"Failed to run pytest: {e}")
            return TestResults(
                success=False,
                passed=0,
                failed=1,
                errors=[{
                    "test": "pytest_execution",
                    "error": str(e),
                    "file": "test_runner.py",
                    "line": 0
                }],
                output=f"pytest execution error: {e}",
                duration=0.0
            )
    
    def run_unit_tests(self) -> TestResults:
        """Run only unit tests."""
        return self.run_pytest("tests/ -m unit")
    
    def run_integration_tests(self) -> TestResults:
        """Run only integration tests."""
        return self.run_pytest("tests/ -m integration")
    
    def run_external_tests(self) -> TestResults:
        """Run external service tests (usually skipped)."""
        return self.run_pytest("tests/ -m external")
    
    def run_specific_tests(self, test_paths: List[str]) -> TestResults:
        """Run specific test files or directories."""
        test_spec = " ".join(test_paths)
        return self.run_pytest(test_spec)
    
    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate that the test environment is properly set up."""
        validation_code = """
import sys
import os
import importlib.util

# Check for required test packages
required_packages = ['pytest', 'pytest_jsonreport']
missing_packages = []

for package in required_packages:
    spec = importlib.util.find_spec(package)
    if spec is None:
        missing_packages.append(package)

# Check for test directory
test_dir_exists = os.path.exists('tests')

# Check for pytest.ini
pytest_ini_exists = os.path.exists('tests/pytest.ini') or os.path.exists('pytest.ini')

# Check for conftest.py
conftest_exists = os.path.exists('tests/conftest.py')

result = {
    'packages_available': len(missing_packages) == 0,
    'missing_packages': missing_packages,
    'test_directory_exists': test_dir_exists,
    'pytest_ini_exists': pytest_ini_exists,
    'conftest_exists': conftest_exists,
    'python_path': sys.path,
    'working_directory': os.getcwd()
}

import json
print(json.dumps(result))
"""
        
        result = self.executor.execute_python(validation_code, "Test environment validation")
        
        if result.success:
            try:
                import json
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {"error": "Could not parse validation results", "output": result.output}
        else:
            return {"error": result.error, "validation_failed": True}
    
    def install_test_dependencies(self) -> bool:
        """Install test dependencies if missing."""
        test_packages = ["pytest", "pytest-json-report", "pytest-cov"]
        
        result = self.executor.install_packages(test_packages)
        if result.success:
            logger.info("✅ Test dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install test dependencies: {result.error}")
            return False
    
    def generate_test_report(self, results: CompositeTestResults) -> str:
        """Generate a detailed test report for Claude Code."""
        if not results.phases:
            return "No test results available"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TEST EXECUTION REPORT")
        report_lines.append("=" * 60)
        
        # Overall summary
        overall_status = "PASSED" if results.overall_success else "FAILED"
        report_lines.append(f"Overall Status: {overall_status}")
        report_lines.append(f"Total Tests: {results.total_passed + results.total_failed}")
        report_lines.append(f"Passed: {results.total_passed}")
        report_lines.append(f"Failed: {results.total_failed}")
        report_lines.append("")
        
        # Phase-by-phase breakdown
        for phase_name, phase_results in results.phases.items():
            report_lines.append(f"📊 {phase_name.upper()} TESTS")
            report_lines.append("-" * 30)
            
            if phase_results.success:
                report_lines.append(f"✅ Status: PASSED")
            else:
                report_lines.append(f"❌ Status: FAILED")
            
            report_lines.append(f"Passed: {phase_results.passed}")
            report_lines.append(f"Failed: {phase_results.failed}")
            report_lines.append(f"Duration: {phase_results.duration:.2f}s")
            
            # Show failures
            if phase_results.errors:
                report_lines.append("\n🔥 FAILURES:")
                for i, error in enumerate(phase_results.errors[:5], 1):  # Show first 5 failures
                    report_lines.append(f"{i}. {error.get('test', 'Unknown test')}")
                    report_lines.append(f"   File: {error.get('file', 'Unknown')}:{error.get('line', 0)}")
                    report_lines.append(f"   Error: {error.get('error', 'Unknown error')[:200]}...")
                
                if len(phase_results.errors) > 5:
                    report_lines.append(f"   ... and {len(phase_results.errors) - 5} more failures")
            
            report_lines.append("")
        
        # Recommendations for failed tests
        if not results.overall_success:
            report_lines.append("💡 RECOMMENDATIONS:")
            report_lines.append("1. Review the failed tests listed above")
            report_lines.append("2. Check recent code changes that might affect these tests")
            report_lines.append("3. Run failing tests locally for detailed debugging")
            
            # Extract unique failing files
            failing_files = set()
            for phase_results in results.phases.values():
                for error in phase_results.errors:
                    if error.get('file'):
                        failing_files.add(error['file'])
            
            if failing_files:
                report_lines.append(f"4. Focus on these files: {', '.join(list(failing_files)[:3])}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
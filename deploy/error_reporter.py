"""
Error reporting module optimized for Claude Code feedback and debugging.
"""

import time
import traceback
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.syntax import Syntax

from .test_runner import CompositeTestResults, TestResults
from .command_executor import ExecutionResult

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ErrorContext:
    """Context information for an error."""
    stage: str
    component: str
    operation: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


class ErrorReporter:
    """Generate structured error reports for Claude Code debugging."""
    
    @staticmethod
    def display_rich_error(title: str, error_msg: str, context: Optional[ErrorContext] = None, 
                          suggestions: Optional[List[str]] = None) -> None:
        """Display rich formatted error using Rich console."""
        # Create error content
        content = Text()
        content.append(error_msg, style="bold red")
        
        # Add context if provided
        if context:
            content.append("\n\nContext:\n", style="bold yellow")
            content.append(f"Stage: {context.stage}\n", style="dim")
            content.append(f"Component: {context.component}\n", style="dim")
            content.append(f"Operation: {context.operation}\n", style="dim")
            if context.file_path:
                content.append(f"File: {context.file_path}:{context.line_number or 'unknown'}\n", style="dim")
        
        # Display error panel
        console.print(Panel(content, title=f"❌ {title}", style="bold red"))
        
        # Display suggestions if provided
        if suggestions:
            suggestion_text = Text()
            for i, suggestion in enumerate(suggestions, 1):
                suggestion_text.append(f"{i}. {suggestion}\n", style="cyan")
            console.print(Panel(suggestion_text, title="💡 Suggestions", style="bold cyan"))
    
    @staticmethod
    def display_test_results_table(results: CompositeTestResults) -> None:
        """Display test results in a Rich table."""
        table = Table(title="🧪 Test Results Summary")
        table.add_column("Phase", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Duration", justify="right", style="magenta")
        
        for phase_name, phase_results in results.phases.items():
            status = "✅" if phase_results.success else "❌"
            table.add_row(
                phase_name,
                status,
                str(phase_results.passed),
                str(phase_results.failed),
                f"{phase_results.duration:.1f}s"
            )
        
        # Add totals row
        table.add_row(
            "TOTAL",
            "✅" if results.overall_success else "❌",
            str(results.total_passed),
            str(results.total_failed),
            f"{results.total_duration:.1f}s",
            style="bold"
        )
        
        console.print(table)
    
    @staticmethod
    def report_test_failures(results: CompositeTestResults) -> str:
        """
        Format test failures for Claude Code with actionable information.
        
        Args:
            results: Combined test results from multiple phases
            
        Returns:
            Formatted error report string
        """
        if results.overall_success:
            return ErrorReporter._format_success_report(results)
        
        report_lines = []
        
        # Header
        report_lines.append("\n" + "=" * 80)
        report_lines.append("🔥 TEST FAILURES DETECTED")
        report_lines.append("=" * 80)
        
        # Summary statistics
        report_lines.append(f"Failed: {results.total_failed} | Passed: {results.total_passed}")
        report_lines.append(f"Phases: {len(results.phases)} | Overall: {'FAILED' if not results.overall_success else 'PASSED'}")
        report_lines.append("")
        
        # Phase-by-phase breakdown
        failed_phases = []
        for phase_name, phase_results in results.phases.items():
            if not phase_results.success:
                failed_phases.append(phase_name)
                report_lines.append(f"❌ {phase_name.upper()} PHASE FAILED")
                report_lines.append(f"   Failed: {phase_results.failed} | Passed: {phase_results.passed}")
                report_lines.append("")
        
        # Detailed failure analysis
        report_lines.append("🔍 DETAILED FAILURE ANALYSIS")
        report_lines.append("-" * 40)
        
        failure_count = 0
        file_error_map = {}
        
        for phase_name, phase_results in results.phases.items():
            if not phase_results.success and phase_results.errors:
                for error in phase_results.errors[:5]:  # Limit to 5 errors per phase
                    failure_count += 1
                    
                    test_name = error.get('test', 'Unknown test')
                    file_path = error.get('file', 'Unknown file')
                    line_num = error.get('line', 0)
                    error_msg = error.get('error', 'Unknown error')
                    
                    # Track errors by file for suggestions
                    if file_path not in file_error_map:
                        file_error_map[file_path] = []
                    file_error_map[file_path].append(error_msg)
                    
                    report_lines.append(f"{failure_count}. 📍 {test_name}")
                    report_lines.append(f"   File: {file_path}:{line_num}")
                    report_lines.append(f"   Phase: {phase_name}")
                    report_lines.append(f"   Error: {error_msg[:150]}...")
                    
                    # Add specific suggestions based on error patterns
                    suggestion = ErrorReporter._get_error_suggestion(error_msg, file_path, phase_name)
                    if suggestion:
                        report_lines.append(f"   💡 Suggestion: {suggestion}")
                    
                    report_lines.append("")
        
        # Action items for Claude Code
        report_lines.append("🚀 CLAUDE CODE ACTION ITEMS")
        report_lines.append("-" * 40)
        
        action_items = ErrorReporter._generate_action_items(failed_phases, file_error_map, results)
        for i, item in enumerate(action_items, 1):
            report_lines.append(f"{i}. {item}")
        
        # Quick commands for local testing
        report_lines.append("")
        report_lines.append("⚡ QUICK LOCAL TEST COMMANDS")
        report_lines.append("-" * 40)
        
        unique_files = list(file_error_map.keys())[:3]  # Top 3 failing files
        if unique_files and unique_files[0] != 'Unknown file':
            report_lines.append(f"# Test specific failing files:")
            for file_path in unique_files:
                if file_path.endswith('.py'):
                    report_lines.append(f"pytest {file_path} -v")
            
            report_lines.append(f"\n# Test by phase:")
            for phase_name in failed_phases:
                report_lines.append(f"pytest tests/ -m {phase_name} -v")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def report_execution_error(result: ExecutionResult, context: ErrorContext) -> str:
        """
        Format execution errors with context for debugging.
        
        Args:
            result: Execution result with error information
            context: Context about where/when the error occurred
            
        Returns:
            Formatted error report string
        """
        report_lines = []
        
        # Header
        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"⚠️  EXECUTION ERROR: {context.stage}")
        report_lines.append("=" * 80)
        
        # Context information
        report_lines.append(f"Stage: {context.stage}")
        report_lines.append(f"Component: {context.component}")
        report_lines.append(f"Operation: {context.operation}")
        if context.file_path:
            report_lines.append(f"File: {context.file_path}:{context.line_number or 'unknown'}")
        report_lines.append(f"Duration: {result.duration:.2f}s")
        report_lines.append("")
        
        # Error details
        report_lines.append("🔥 ERROR DETAILS")
        report_lines.append("-" * 20)
        report_lines.append(f"Error: {result.error}")
        
        if result.output and result.output.strip():
            report_lines.append("\n📄 OUTPUT")
            report_lines.append("-" * 20)
            # Show last 500 chars of output to avoid overwhelming
            output_preview = result.output[-500:] if len(result.output) > 500 else result.output
            report_lines.append(output_preview)
        
        # Suggestions
        if context.suggestion:
            report_lines.append(f"\n💡 SUGGESTION")
            report_lines.append("-" * 20)
            report_lines.append(context.suggestion)
        
        # Auto-generated suggestions based on error patterns
        auto_suggestion = ErrorReporter._get_execution_error_suggestion(result.error, context)
        if auto_suggestion:
            report_lines.append(f"\n🤖 AUTO-SUGGESTION")
            report_lines.append("-" * 20)
            report_lines.append(auto_suggestion)
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def report_deployment_failure(stage: str, details: Dict[str, Any]) -> str:
        """
        Format deployment failures with specific guidance.
        
        Args:
            stage: The deployment stage that failed
            details: Error details from the failed stage
            
        Returns:
            Formatted error report string
        """
        report_lines = []
        
        # Header
        report_lines.append("\n" + "🚨" * 20)
        report_lines.append(f"DEPLOYMENT FAILED AT: {stage.upper()}")
        report_lines.append("🚨" * 20)
        
        # Error information
        error_msg = details.get('error', 'Unknown error')
        report_lines.append(f"Error: {error_msg}")
        
        if 'traceback' in details:
            report_lines.append(f"\nTraceback:\n{details['traceback']}")
        
        # Stage-specific guidance
        guidance = ErrorReporter._get_stage_specific_guidance(stage, details)
        if guidance:
            report_lines.append(f"\n🎯 STAGE-SPECIFIC GUIDANCE")
            report_lines.append("-" * 30)
            report_lines.append(guidance)
        
        # Recovery steps
        recovery_steps = ErrorReporter._get_recovery_steps(stage, details)
        if recovery_steps:
            report_lines.append(f"\n🔄 RECOVERY STEPS")
            report_lines.append("-" * 30)
            for i, step in enumerate(recovery_steps, 1):
                report_lines.append(f"{i}. {step}")
        
        report_lines.append("\n" + "🚨" * 20)
        
        return "\n".join(report_lines)
    
    @staticmethod
    def _format_success_report(results: CompositeTestResults) -> str:
        """Format success report when all tests pass."""
        return f"""
✅ ALL TESTS PASSED!

Summary: {results.total_passed} tests passed across {len(results.phases)} phases
{results.get_summary()}

Ready for deployment! 🚀
"""
    
    @staticmethod
    def _get_error_suggestion(error_msg: str, file_path: str, phase: str) -> Optional[str]:
        """Generate specific suggestions based on error patterns."""
        error_lower = error_msg.lower()
        
        # Import errors
        if "import" in error_lower and ("no module" in error_lower or "cannot import" in error_lower):
            return "Check package dependencies in requirements.txt or install missing packages"
        
        # Configuration errors
        if "config" in error_lower or "yaml" in error_lower:
            return "Verify agent_config.yaml exists and has correct format"
        
        # API key errors
        if "api" in error_lower and ("key" in error_lower or "authentication" in error_lower):
            return "Check API key configuration in Databricks secrets or environment variables"
        
        # Test-specific errors
        if phase == "unit":
            return "Run this test locally to get more detailed error information"
        elif phase == "integration":
            return "Check external service connectivity and configuration"
        
        # File path errors
        if "file" in error_lower or "path" in error_lower:
            return "Verify file paths are correct relative to workspace root"
        
        return None
    
    @staticmethod
    def _get_execution_error_suggestion(error: str, context: ErrorContext) -> Optional[str]:
        """Generate suggestions for execution errors."""
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "Increase timeout value or check for infinite loops in the code"
        
        if "cluster" in error_lower:
            return "Verify cluster is running and accessible. Check cluster_id in configuration"
        
        if "context" in error_lower:
            return "Command execution context may have been lost. This usually resolves on retry"
        
        if "permission" in error_lower:
            return "Check workspace permissions and profile authentication"
        
        if context.stage == "model_logging":
            return "Ensure MLflow is properly configured and Unity Catalog is accessible"
        
        if context.stage == "endpoint_deployment":
            return "Check serving endpoint quotas and model registration status"
        
        return None
    
    @staticmethod
    def _generate_action_items(failed_phases: List[str], file_error_map: Dict[str, List[str]], 
                             results: CompositeTestResults) -> List[str]:
        """Generate specific action items for Claude Code."""
        actions = []
        
        # Phase-specific actions
        if "unit" in failed_phases:
            actions.append("Fix unit test failures - these are basic functionality issues")
        
        if "integration" in failed_phases:
            actions.append("Review integration test setup - check external dependencies")
        
        # File-specific actions
        failing_files = [f for f in file_error_map.keys() if f != 'Unknown file']
        if failing_files:
            actions.append(f"Focus on these files: {', '.join(failing_files[:3])}")
        
        # Pattern-based actions
        all_errors = []
        for phase_results in results.phases.values():
            for error in phase_results.errors:
                all_errors.append(error.get('error', '').lower())
        
        error_text = ' '.join(all_errors)
        
        if 'import' in error_text:
            actions.append("Check and fix import statements and dependencies")
        
        if 'config' in error_text:
            actions.append("Verify configuration files are present and valid")
        
        if 'api' in error_text and 'key' in error_text:
            actions.append("Check API key configuration in secrets")
        
        # Always add these general actions
        actions.append("Run failing tests locally for detailed debugging")
        actions.append("Check recent changes that might have broken these tests")
        
        return actions
    
    @staticmethod
    def _get_stage_specific_guidance(stage: str, details: Dict[str, Any]) -> Optional[str]:
        """Get specific guidance based on deployment stage."""
        if stage == "file_sync":
            return ("Check Databricks CLI configuration and workspace permissions. "
                   "Ensure profile is set up correctly.")
        
        elif stage == "test_execution":
            return ("Tests failed on Databricks. Run tests locally first to verify they pass. "
                   "Check test dependencies and environment setup.")
        
        elif stage == "model_logging":
            return ("MLflow logging failed. Verify Unity Catalog access and model configuration. "
                   "Check that agent code can be imported properly.")
        
        elif stage == "model_registration":
            return ("Model registration in Unity Catalog failed. Check catalog/schema permissions "
                   "and ensure model was logged successfully.")
        
        elif stage == "agent_deployment":
            return ("Agent Framework deployment failed. Check serving endpoint quotas and "
                   "model registry status. Verify environment variables are set correctly.")
        
        elif stage == "endpoint_validation":
            return ("Deployed endpoint validation failed. Endpoint may still be starting up, "
                   "or there may be runtime configuration issues.")
        
        return None
    
    @staticmethod
    def _get_recovery_steps(stage: str, details: Dict[str, Any]) -> List[str]:
        """Get recovery steps based on failure stage."""
        steps = []
        
        if stage in ["test_execution", "file_sync"]:
            steps.extend([
                "Fix the identified issues in the code",
                "Run tests locally to verify fixes",
                "Re-run deployment with same command"
            ])
        
        elif stage in ["model_logging", "model_registration"]:
            steps.extend([
                "Check Unity Catalog permissions",
                "Verify agent code can be imported locally",
                "Re-run deployment from model logging stage"
            ])
        
        elif stage == "agent_deployment":
            steps.extend([
                "Check serving endpoint quotas in workspace",
                "Verify model is registered and accessible",
                "Consider using different endpoint name if conflicts exist"
            ])
        
        elif stage == "endpoint_validation":
            steps.extend([
                "Wait 5-10 minutes for endpoint to fully initialize",
                "Check endpoint status in Databricks UI",
                "Test endpoint manually with simple query"
            ])
        
        # Always add general recovery steps
        steps.extend([
            "Check Databricks workspace for any error details",
            "Review deployment logs for additional context"
        ])
        
        return steps
    
    @staticmethod
    def create_summary_report(success: bool, stage: str, duration: float, 
                            details: Optional[Dict[str, Any]] = None) -> str:
        """Create a concise summary report for deployment results."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        
        report = f"""
{status} - Deployment {stage}
Duration: {duration:.2f}s
"""
        
        if not success and details:
            if 'error' in details:
                report += f"Error: {details['error'][:100]}...\n"
            if 'stage' in details:
                report += f"Failed at: {details['stage']}\n"
        
        return report
"""
Command-line interface for the deployment package.
"""

import argparse
import logging
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm

console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Deploy LangGraph Research Agent to Databricks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard deployment with tests
  python -m deploy --env dev

  # Deploy with custom endpoint name
  python -m deploy --env dev --endpoint-name "my-agent-v2"

  # Delete existing endpoint before deployment
  python -m deploy --env dev --delete-existing-endpoint

  # Clean workspace and deploy fresh
  python -m deploy --env dev --clean-workspace --delete-existing-endpoint

  # Quick deployment without tests (default)
  python -m deploy --env dev

  # Deployment with tests
  python -m deploy --env dev --run-tests

  # Test-only mode (no deployment)
  python -m deploy --env dev --run-tests --test-only

  # Run specific test markers
  python -m deploy --env dev --run-tests --test-markers "unit"
  python -m deploy --env dev --run-tests --test-markers "integration"
  python -m deploy --env dev --run-tests --test-markers "not external"

  # Dry run to see what would be done
  python -m deploy --env dev --dry-run

  # Verbose output for debugging
  python -m deploy --env dev --verbose

  # Production deployment (typically uses different compute)
  python -m deploy --env prod

Environment Examples:
  - dev: Development environment with small compute
  - staging: Staging environment for testing
  - prod: Production environment with high-performance compute
  - test: Test environment with serverless compute
        """
    )
    
    # Environment selection
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod", "test"],
        default="dev",
        help="Target deployment environment (default: dev)"
    )
    
    # Endpoint configuration
    parser.add_argument(
        "--endpoint-name",
        help="Override the default endpoint name from configuration"
    )
    
    parser.add_argument(
        "--delete-existing-endpoint",
        action="store_true",
        help="Delete existing endpoint before deployment (if it exists)"
    )
    
    # Workspace management
    parser.add_argument(
        "--clean-workspace",
        action="store_true",
        help="Remove existing deployment folder in workspace before sync"
    )
    
    # Testing options
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run tests before deployment (default: tests are skipped)"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run tests only, do not proceed with deployment"
    )
    
    parser.add_argument(
        "--test-markers",
        help="Pytest markers to filter tests (e.g., 'unit', 'integration', 'not external')"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Stop on first test failure (default: True)"
    )
    
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help="Continue testing even if some tests fail"
    )
    
    # Deployment control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing deployment"
    )
    
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync files to workspace, do not run tests or deployment"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts for destructive operations"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default="conf/deploy/",
        help="Path to deployment configuration directory or file (default: conf/deploy/)"
    )
    
    parser.add_argument(
        "--cluster-id",
        help="Override cluster ID for command execution (overrides config)"
    )
    
    # Output and logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output and detailed logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    # Validation options
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip endpoint validation tests after deployment"
    )
    
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=300,
        help="Timeout for endpoint validation in seconds (default: 300)"
    )
    
    # Endpoint management
    parser.add_argument(
        "--list-endpoints",
        action="store_true",
        help="List existing serving endpoints and exit"
    )
    
    parser.add_argument(
        "--endpoint-status",
        help="Check status of a specific endpoint and exit"
    )
    
    parser.add_argument(
        "--cleanup-old-endpoints",
        help="Clean up old endpoints matching pattern (e.g., 'dev-', 'test-')"
    )
    
    parser.add_argument(
        "--cleanup-max-age",
        type=int,
        default=24,
        help="Maximum age in hours for endpoint cleanup (default: 24)"
    )
    
    # Advanced options
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Overall deployment timeout in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--retry-count",
        type=int,
        default=1,
        help="Number of retry attempts for failed operations (default: 1)"
    )
    
    return parser


def parse_arguments(args: List[str] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Optional list of arguments to parse (for testing)
        
    Returns:
        Parsed arguments namespace
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Validate argument combinations
    _validate_arguments(parsed_args)
    
    return parsed_args


def _validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate argument combinations and set up derived values.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        SystemExit: If invalid argument combinations are found
    """
    # Mutual exclusions
    if args.test_only and not args.run_tests:
        print("Error: --test-only requires --run-tests flag")
        exit(1)
    
    if args.sync_only and (args.test_only or args.delete_existing_endpoint):
        print("Error: --sync-only cannot be combined with test or deployment options")
        exit(1)
    
    if args.verbose and args.quiet:
        print("Error: --verbose and --quiet are mutually exclusive")
        exit(1)
    
    if args.dry_run and args.test_only:
        print("Error: --dry-run and --test-only are mutually exclusive")
        exit(1)
    
    # Set up logging level based on verbose/quiet flags
    if args.verbose:
        args.log_level = "DEBUG"
    elif args.quiet:
        args.log_level = "WARNING"


def configure_logging(args: argparse.Namespace) -> None:
    """
    Configure logging based on command-line arguments.
    
    Args:
        args: Parsed arguments namespace
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.verbose:
        log_format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    elif args.quiet:
        log_format = '%(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # Adjust external library log levels
    if not args.verbose:
        logging.getLogger('databricks').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_deployment_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Extract deployment configuration from parsed arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Returns:
        Dictionary with deployment configuration
    """
    return {
        "environment": args.env,
        "config_file": args.config,
        "endpoint_name": args.endpoint_name,
        "delete_existing_endpoint": args.delete_existing_endpoint,
        "clean_workspace": args.clean_workspace,
        "run_tests": args.run_tests,
        "test_only": args.test_only,
        "test_markers": args.test_markers,
        "fail_fast": args.fail_fast,
        "dry_run": args.dry_run,
        "sync_only": args.sync_only,
        "force": args.force,
        "cluster_id": args.cluster_id,
        "verbose": args.verbose,
        "quiet": args.quiet,
        "skip_validation": args.skip_validation,
        "validation_timeout": args.validation_timeout,
        "timeout": args.timeout,
        "retry_count": args.retry_count,
        
        # Management operations
        "list_endpoints": args.list_endpoints,
        "endpoint_status": args.endpoint_status,
        "cleanup_old_endpoints": args.cleanup_old_endpoints,
        "cleanup_max_age": args.cleanup_max_age
    }


def print_configuration_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the deployment configuration.
    
    Args:
        config: Deployment configuration dictionary
    """
    print("=" * 60)
    print("DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {config['environment']}")
    print(f"Config File: {config['config_file']}")
    
    if config.get('endpoint_name'):
        print(f"Endpoint Name: {config['endpoint_name']} (override)")
    
    if config.get('cluster_id'):
        print(f"Cluster ID: {config['cluster_id']} (override)")
    
    # Deployment flags
    flags = []
    if config['clean_workspace']:
        flags.append("clean-workspace")
    if config['delete_existing_endpoint']:
        flags.append("delete-existing-endpoint")
    if config['run_tests']:
        flags.append("run-tests")
    if config['test_only']:
        flags.append("test-only")
    if config['dry_run']:
        flags.append("dry-run")
    if config['sync_only']:
        flags.append("sync-only")
    if config['force']:
        flags.append("force")
    
    if flags:
        print(f"Flags: {', '.join(flags)}")
    
    if config.get('test_markers'):
        print(f"Test Markers: {config['test_markers']}")
    
    print("=" * 60)
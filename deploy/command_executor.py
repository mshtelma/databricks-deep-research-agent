"""
Command Execution API wrapper for running Python code on Databricks clusters.
"""

import time
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute
from databricks.sdk.service.compute import State
from databricks.sdk.core import Config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.status import Status

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ExecutionResult:
    """Result of command execution on Databricks cluster."""
    success: bool
    output: str
    error: Optional[str] = None
    duration: float = 0.0
    command_id: Optional[str] = None
    status: Optional[str] = None


@dataclass 
class TestResults:
    """Results of pytest execution."""
    success: bool
    passed: int
    failed: int
    errors: list
    output: str
    duration: float = 0.0
    
    @classmethod
    def from_json(cls, json_output: str) -> "TestResults":
        """Parse pytest results from JSON output."""
        try:
            data = json.loads(json_output)
            return cls(
                success=data.get('success', False),
                passed=data.get('passed', 0),
                failed=data.get('failed', 0),
                errors=data.get('errors', []),
                output=data.get('stdout', '') + data.get('stderr', ''),
                duration=data.get('duration', 0.0)
            )
        except json.JSONDecodeError as e:
            return cls(
                success=False,
                passed=0,
                failed=1,
                errors=[{"test": "json_parse", "error": f"Failed to parse results: {e}"}],
                output=json_output,
                duration=0.0
            )


class CommandExecutionError(Exception):
    """Exception raised when command execution fails."""
    pass


class CommandExecutor:
    """Execute Python code on Databricks clusters using Command Execution API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize command executor with cluster configuration."""
        self.config = config
        self.workspace_client = self._create_workspace_client()
        self.cluster_id = self._get_cluster_id()
        self.context_id: Optional[str] = None
        
        logger.info(f"CommandExecutor initialized with cluster: {self.cluster_id}")
    
    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client with profile support."""
        profile = self.config.get("profile")
        if profile:
            logger.info(f"Using Databricks profile: {profile}")
            config = Config(profile=profile)
            return WorkspaceClient(config=config)
        else:
            logger.info("Using environment variable authentication")
            return WorkspaceClient()
    
    def _get_cluster_id(self) -> str:
        """Get cluster ID from configuration."""
        # Check command_execution specific config first
        cluster_id = self.config.get("command_execution", {}).get("cluster_id")
        
        # Fallback to job config for backwards compatibility
        if not cluster_id:
            cluster_id = self.config.get("job", {}).get("existing_cluster_id")
        
        if not cluster_id:
            raise CommandExecutionError(
                "No cluster_id found in configuration. "
                "Add 'command_execution.cluster_id' or 'job.existing_cluster_id'"
            )
        
        return cluster_id
    
    def _get_cluster_state(self) -> Optional[State]:
        """Check current cluster state."""
        try:
            cluster = self.workspace_client.clusters.get(self.cluster_id)
            return cluster.state
        except Exception as e:
            logger.warning(f"Failed to get cluster state: {e}")
            return None
    
    def _should_auto_start_cluster(self) -> bool:
        """Check if cluster auto-start is enabled."""
        return self.config.get("command_execution", {}).get("auto_start", True)
    
    def _get_cluster_start_timeout(self) -> int:
        """Get timeout for cluster start operations."""
        return self.config.get("command_execution", {}).get("cluster_start_timeout", 600)
    
    def _start_cluster(self) -> bool:
        """Start the cluster."""
        try:
            console.print(f"[yellow]🚀 Starting cluster {self.cluster_id}...")
            self.workspace_client.clusters.start(self.cluster_id)
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to start cluster: {e}")
            logger.error(f"Failed to start cluster: {e}")
            return False
    
    def _wait_for_cluster_running(self, timeout: int = 600) -> bool:
        """Wait for cluster to reach RUNNING state with Rich progress."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Starting cluster...", total=None)
            
            start_time = time.time()
            last_state = None
            
            while time.time() - start_time < timeout:
                current_state = self._get_cluster_state()
                
                # Update progress description when state changes
                if current_state != last_state:
                    if current_state == State.RUNNING:
                        progress.update(task, description="[green]✅ Cluster is running!")
                        console.print("[bold green]✅ Cluster started successfully!")
                        return True
                    else:
                        progress.update(task, description=f"[cyan]Cluster state: {current_state}")
                    last_state = current_state
                
                time.sleep(5)  # Check every 5 seconds
            
            console.print(f"[bold red]❌ Cluster failed to start within {timeout}s timeout")
            return False
    
    def _ensure_cluster_running(self) -> bool:
        """Ensure cluster is in RUNNING state, start if needed."""
        with console.status("[bold blue]Checking cluster state...") as status:
            current_state = self._get_cluster_state()
            
            if current_state is None:
                console.print("[red]❌ Could not determine cluster state")
                return False
            
            if current_state == State.RUNNING:
                console.print("[green]✅ Cluster is already running")
                return True
            
            console.print(f"[yellow]📊 Cluster state: {current_state}")
            
            if current_state in [State.TERMINATED, State.TERMINATING]:
                if not self._should_auto_start_cluster():
                    console.print("[red]❌ Cluster is stopped and auto-start is disabled")
                    return False
                
                console.print("[blue]🚀 Auto-start enabled, starting cluster...")
                if not self._start_cluster():
                    return False
                
                timeout = self._get_cluster_start_timeout()
                return self._wait_for_cluster_running(timeout)
                
            elif current_state in [State.PENDING, State.RESTARTING, State.RESIZING]:
                console.print(f"[blue]⏳ Waiting for cluster to be ready (current: {current_state})")
                timeout = self._get_cluster_start_timeout()
                return self._wait_for_cluster_running(timeout)
            
            else:
                console.print(f"[red]❌ Cluster is in unexpected state: {current_state}")
                return False
    
    def _ensure_context(self) -> str:
        """Ensure cluster is running and execution context exists."""
        if not self.context_id:
            # First ensure cluster is running
            if not self._ensure_cluster_running():
                raise CommandExecutionError(
                    f"Cluster {self.cluster_id} is not available. "
                    f"Check cluster state in Databricks UI."
                )
            
            # Then create context with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with console.status("[bold blue]Creating execution context..."):
                        context = self.workspace_client.command_execution.create(
                            cluster_id=self.cluster_id,
                            language=compute.Language.PYTHON
                        ).result()
                        self.context_id = context.id
                    
                    console.print(f"[green]✅ Created execution context: {self.context_id}")
                    logger.info(f"Created execution context: {self.context_id}")
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        console.print(f"[red]❌ Failed to create execution context after {max_retries} attempts: {e}")
                        raise CommandExecutionError(
                            f"Failed to create execution context after {max_retries} attempts: {e}"
                        )
                    else:
                        console.print(f"[yellow]⚠️  Attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(10)
        
        return self.context_id
    
    def execute_python(self, code: str, description: str = "", timeout: int = 300) -> ExecutionResult:
        """
        Execute Python code on the Databricks cluster.
        
        Args:
            code: Python code to execute
            description: Description for logging
            timeout: Timeout in seconds
            
        Returns:
            ExecutionResult with success status, output, and error info
        """
        start_time = time.time()
        context_id = self._ensure_context()
        
        if description:
            console.print(f"[blue]🔧 {description}")
            logger.info(f"Executing: {description}")
        
        try:
            # Execute command
            with console.status(f"[bold blue]Executing: {description or 'Python code'}..."):
                command_result = self.workspace_client.command_execution.execute(
                    cluster_id=self.cluster_id,
                    context_id=context_id,
                    language=compute.Language.PYTHON,
                    command=code
                ).result()
            
            # Wait for completion and get result
            command_id = command_result.id
            result = self._wait_for_completion(command_id, timeout, description)
            result.duration = time.time() - start_time
            
            if result.success and description:
                console.print(f"[green]✅ {description} completed in {result.duration:.2f}s")
                logger.info(f"✅ {description} completed in {result.duration:.2f}s")
            elif not result.success and description:
                console.print(f"[red]❌ {description} failed: {result.error}")
                logger.error(f"❌ {description} failed: {result.error}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            
            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                duration=duration
            )
    
    def _wait_for_completion(self, command_id: str, timeout: int, description: str = "") -> ExecutionResult:
        """Wait for command to complete and return results with Rich progress."""
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"[cyan]Executing: {description or 'command'}...", total=None)
            
            while time.time() - start_time < timeout:
                try:
                    status = self.workspace_client.command_execution.command_status(
                        cluster_id=self.cluster_id,
                        context_id=self.context_id,
                        command_id=command_id
                    )
                    
                    if status.status == compute.CommandStatus.FINISHED:
                        # Command completed successfully
                        output = ""
                        error = None
                        
                        if status.results and status.results.data:
                            output = status.results.data
                        
                        progress.update(task, description=f"[green]✅ Completed: {description or 'command'}")
                        return ExecutionResult(
                            success=True,
                            output=output,
                            error=error,
                            command_id=command_id,
                            status="FINISHED"
                        )
                    
                    elif status.status == compute.CommandStatus.ERROR:
                        # Command failed
                        error_msg = "Command execution error"
                        if status.results and status.results.data:
                            error_msg = status.results.data
                        
                        progress.update(task, description=f"[red]❌ Failed: {description or 'command'}")
                        return ExecutionResult(
                            success=False,
                            output="",
                            error=error_msg,
                            command_id=command_id,
                            status="ERROR"
                        )
                    
                    elif status.status == compute.CommandStatus.CANCELLED:
                        progress.update(task, description=f"[yellow]⚠️  Cancelled: {description or 'command'}")
                        return ExecutionResult(
                            success=False,
                            output="",
                            error="Command was cancelled",
                            command_id=command_id,
                            status="CANCELLED"
                        )
                    
                    # Still running, update progress
                    elapsed = time.time() - start_time
                    progress.update(task, description=f"[cyan]Running: {description or 'command'} ({elapsed:.0f}s)")
                    
                    # Still running, wait a bit more
                    time.sleep(2)
                    
                except Exception as e:
                    progress.update(task, description=f"[red]❌ Status error: {description or 'command'}")
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=f"Failed to get command status: {e}",
                        command_id=command_id
                    )
            
            # Timeout
            progress.update(task, description=f"[red]⏰ Timeout: {description or 'command'}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                command_id=command_id,
                status="TIMEOUT"
            )
    
    def run_pytest(self, test_spec: str, markers: Optional[str] = None) -> TestResults:
        """
        Run pytest on Databricks and return parsed results.
        
        Args:
            test_spec: Test specification (e.g., "tests/" or "tests/test_file.py")
            markers: Optional pytest markers to filter tests
            
        Returns:
            TestResults with parsed pytest output
        """
        pytest_cmd = f"pytest {test_spec}"
        if markers:
            pytest_cmd += f" -m '{markers}'"
        
        # Add JSON reporting for easier parsing
        pytest_cmd += " --json-report --json-report-file=test_results.json -v"
        
        code = f"""
import subprocess
import json
import sys
import os
import time

# Print environment info for debugging
print("🔍 Test Environment Info:")
print(f"Current directory: {{os.getcwd()}}")
print(f"Python executable: {{sys.executable}}")
print(f"Python path: {{':'.join(sys.path[:3])}}...")

# Check if tests directory exists
if os.path.exists('tests'):
    test_files = [f for f in os.listdir('tests') if f.startswith('test_') and f.endswith('.py')]
    print(f"📁 Found {{len(test_files)}} test files in tests/: {{test_files[:3]}}")
else:
    print("❌ No 'tests' directory found!")

# Install pytest-json-report if not available
print("📦 Installing pytest-json-report...")
try:
    import pytest_jsonreport
    print("✅ pytest-json-report already available")
except ImportError:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest-json-report'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ pytest-json-report installed")
    else:
        print(f"❌ Failed to install pytest-json-report: {{result.stderr}}")

# Build and display pytest command
cmd = ['python', '-m', 'pytest', '{test_spec}', '--json-report', '--json-report-file=test_results.json', '-v', '--tb=short']
if '{markers}':
    cmd.extend(['-m', '{markers}'])

print(f"🚀 Running command: {{' '.join(cmd)}}")

# Run pytest
start_time = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

duration = time.time() - start_time

print(f"🏁 Pytest completed in {{duration:.1f}}s")
print(f"Return code: {{result.returncode}}")

# Show pytest stdout/stderr for debugging
if result.stdout:
    print("📄 Pytest STDOUT:")
    print(result.stdout)
if result.stderr:
    print("📄 Pytest STDERR:")  
    print(result.stderr)

# Parse JSON report if available
try:
    if os.path.exists('test_results.json'):
        print("✅ JSON report file found")
        with open('test_results.json', 'r') as f:
            test_data = json.load(f)
        
        print(f"📊 Test summary from JSON: {{test_data.get('summary', dict())}}")
        
        # Extract failed test details
        errors = []
        for test in test_data.get('tests', []):
            if test.get('outcome') == 'failed':
                errors.append({{
                    'test': test.get('nodeid', 'unknown'),
                    'file': test.get('location', ['unknown'])[0],
                    'line': test.get('location', [None, 0])[1] or 0,
                    'error': test.get('call', dict()).get('longrepr', 'Unknown error'),
                    'message': str(test.get('call', dict()).get('longrepr', ''))[:500]
                }})
        
        # Format output
        output = {{
            'success': result.returncode == 0,
            'passed': test_data.get('summary', dict()).get('passed', 0),
            'failed': test_data.get('summary', dict()).get('failed', 0),
            'errors': errors,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration
        }}
    else:
        print("❌ JSON report file not found, using fallback parsing")
        # Fallback if JSON report not available
        output = {{
            'success': result.returncode == 0,
            'passed': 0,
            'failed': 1 if result.returncode != 0 else 0,
            'errors': [{{'test': 'pytest_run', 'error': result.stderr or result.stdout}}],
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration
        }}
        
except Exception as parse_error:
    print(f"❌ Error parsing results: {{parse_error}}")
    output = {{
        'success': False,
        'passed': 0,
        'failed': 1,
        'errors': [{{'test': 'result_parsing', 'error': str(parse_error)}}],
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration': duration
    }}

print("📤 Returning result:")
print(json.dumps(output, indent=2))
"""
        
        # Log the command being executed for visibility
        console.print(f"[cyan]🧪 Executing pytest: {test_spec}")
        if markers:
            console.print(f"[dim]   with markers: {markers}")
        
        result = self.execute_python(
            code,
            description=f"Running pytest: {test_spec}" + (f" with markers: {markers}" if markers else ""),
            timeout=600  # Pytest can take a while
        )
        
        if not result.success:
            # Command execution failed
            console.print(f"[red]❌ Pytest execution failed: {result.error}")
            if result.output:
                console.print(f"[dim]Raw output: {result.output[:500]}...")
            return TestResults(
                success=False,
                passed=0,
                failed=1,
                errors=[{
                    "test": "command_execution",
                    "error": result.error or "Unknown execution error",
                    "file": "command_executor.py",
                    "line": 0
                }],
                output=result.output,
                duration=result.duration
            )
        
        # Try to parse JSON, but display raw output on failure
        try:
            return TestResults.from_json(result.output)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Failed to parse pytest JSON output: {e}")
            console.print(f"[yellow]📄 Raw pytest output:")
            console.print(f"[dim]{result.output}")
            
            # Try to extract basic info from raw output
            output_lines = result.output.split('\n')
            failed_count = 0
            passed_count = 0
            
            for line in output_lines:
                # Look for patterns like:
                # "1 failed, 0 passed in 0.05s"
                # "= 1 failed, 0 passed in 0.05s ="
                # "FAILED (failures=1)"
                if 'failed' in line.lower():
                    if 'passed' in line.lower():
                        match = re.search(r'(\d+)\s+failed.*?(\d+)\s+passed', line)
                        if match:
                            failed_count = int(match.group(1))
                            passed_count = int(match.group(2))
                            break
                    else:
                        # Just failed count
                        match = re.search(r'(\d+)\s+failed', line)
                        if match:
                            failed_count = int(match.group(1))
                elif 'FAILED' in line and 'failures=' in line:
                    match = re.search(r'failures=(\d+)', line)
                    if match:
                        failed_count = int(match.group(1))
            
            return TestResults(
                success=False,
                passed=passed_count,
                failed=failed_count if failed_count > 0 else 1,
                errors=[{"test": "json_parse", "error": f"Failed to parse JSON: {e}"}],
                output=result.output,
                duration=result.duration
            )
    
    def install_packages(self, packages: list, timeout: int = 600) -> ExecutionResult:
        """Install Python packages on the cluster."""
        packages_str = " ".join(packages)
        code = f"""
import subprocess
import sys

# Install packages
result = subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--upgrade'
] + {repr(packages)},
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"Successfully installed: {packages_str}")
else:
    print(f"Installation failed: {{result.stderr}}")
    exit(result.returncode)
"""
        
        return self.execute_python(
            code,
            description=f"Installing packages: {packages_str}",
            timeout=timeout
        )
    
    def cleanup(self):
        """Clean up execution context."""
        if self.context_id:
            try:
                self.workspace_client.command_execution.destroy(
                    cluster_id=self.cluster_id,
                    context_id=self.context_id
                )
                logger.info(f"Cleaned up execution context: {self.context_id}")
                self.context_id = None
            except Exception as e:
                logger.warning(f"Failed to cleanup context: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
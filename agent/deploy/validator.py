"""
Endpoint validation and testing module.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .command_executor import CommandExecutor, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of endpoint validation tests."""
    success: bool
    tests_passed: int
    tests_total: int
    endpoint_name: str
    endpoint_url: Optional[str] = None
    results: List[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0
    
    def __post_init__(self):
        """Initialize results list if None."""
        if self.results is None:
            self.results = []
    
    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.success
    
    @classmethod
    def from_json(cls, json_output: str, endpoint_name: str) -> "ValidationResult":
        """Parse validation results from JSON output."""
        try:
            data = json.loads(json_output)
            
            # Extract error message if validation failed
            error = data.get('error')
            if not data.get('success', False) and not error:
                # Try to extract error from results
                results = data.get('results', [])
                if results:
                    failed_tests = [r for r in results if not r.get('success', False)]
                    if failed_tests:
                        errors = [f"{t.get('name', 'Test')}: {t.get('error', 'Failed')}" 
                                 for t in failed_tests[:3]]  # Show first 3 failures
                        error = f"Validation failed - {len(failed_tests)}/{len(results)} tests failed. " + "; ".join(errors)
                    else:
                        error = f"Validation failed but no specific test errors found"
                else:
                    error = f"Validation failed - no test results available (endpoint may not be ready)"
            
            return cls(
                success=data.get('success', False),
                tests_passed=data.get('tests_passed', 0),
                tests_total=data.get('tests_total', 0),
                endpoint_name=endpoint_name,
                endpoint_url=data.get('endpoint_url'),
                results=data.get('results', []),
                error=error,
                duration=data.get('duration', 0.0)
            )
        except json.JSONDecodeError as e:
            return cls(
                success=False,
                tests_passed=0,
                tests_total=1,
                endpoint_name=endpoint_name,
                error=f"Failed to parse validation results: {e}",
                results=[{"error": f"JSON parse error: {e}"}]
            )


class Validator:
    """Validate deployed endpoints and run acceptance tests."""
    
    def __init__(self, executor: CommandExecutor):
        """Initialize validator with command executor."""
        self.executor = executor
    
    def validate_endpoint(self, endpoint_name: str, config: Dict[str, Any], 
                         timeout: int = 300) -> ValidationResult:
        """
        Run comprehensive validation tests against deployed endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to validate
            config: Deployment configuration
            timeout: Timeout for validation tests
            
        Returns:
            ValidationResult with test outcomes
        """
        logger.info(f"🧪 Validating endpoint: {endpoint_name}")
        
        validation_code = f'''
import requests
import json
import time
from databricks.sdk import WorkspaceClient

# Initialize workspace client
workspace_client = WorkspaceClient()

start_time = time.time()

try:
    # Get endpoint information
    endpoint = workspace_client.serving_endpoints.get("{endpoint_name}")
    endpoint_url = f"{{workspace_client.config.host}}/serving-endpoints/{endpoint_name}/invocations"
    
    # Test queries with different complexity levels
    test_queries = [
        {{
            "name": "Health Check",
            "query": "Hello, are you there?",
            "type": "warmup",
            "timeout": 20,
            "description": "Basic connectivity test"
        }},
        {{
            "name": "Simple Math",
            "query": "What is 42 + 58?",
            "type": "simple",
            "timeout": 30,
            "description": "Tests basic reasoning"
        }},
        {{
            "name": "Research Query", 
            "query": "What are the key benefits of using LangGraph for building AI agents? Provide 3 bullet points.",
            "type": "research",
            "timeout": 60,
            "description": "Tests research and synthesis capabilities"
        }},
        {{
            "name": "Code Understanding",
            "query": "Explain what this Python code does: lambda x: x**2 if x > 0 else 0",
            "type": "code",
            "timeout": 45,
            "description": "Tests code analysis abilities"
        }}
    ]
    
    results = []
    successful_tests = 0
    
    for test_query in test_queries:
        try:
            query_start = time.time()
            
            payload = {{
                "input": [{{"role": "user", "content": test_query["query"]}}]
            }}
            
            # Make authenticated request
            response = requests.post(
                endpoint_url,
                json=payload,
                headers={{
                    "Authorization": f"Bearer {{workspace_client.config.token}}",
                    "Content-Type": "application/json"
                }},
                timeout=test_query["timeout"]
            )
            
            query_duration = time.time() - query_start
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    # Check if response has expected structure
                    if "output" in response_data and response_data["output"]:
                        successful_tests += 1
                        results.append({{
                            "name": test_query["name"],
                            "query": test_query["query"],
                            "type": test_query["type"],
                            "description": test_query.get("description", ""),
                            "status_code": response.status_code,
                            "success": True,
                            "response_time": query_duration,
                            "response_length": len(str(response_data.get("output", ""))),
                            "has_citations": "citations" in str(response_data).lower()
                        }})
                    else:
                        results.append({{
                            "name": test_query["name"],
                            "query": test_query["query"],
                            "type": test_query["type"],
                            "success": False,
                            "error": "Empty or invalid response structure",
                            "response_time": query_duration,
                            "status_code": response.status_code
                        }})
                except json.JSONDecodeError:
                    results.append({{
                        "name": test_query["name"],
                        "query": test_query["query"],
                        "type": test_query["type"],
                        "success": False,
                        "error": "Invalid JSON response",
                        "response_time": query_duration,
                        "status_code": response.status_code
                    }})
            else:
                results.append({{
                    "name": test_query["name"],
                    "query": test_query["query"],
                    "type": test_query["type"],
                    "status_code": response.status_code,
                    "success": False,
                    "error": response.text[:200] if response.text else "HTTP error",
                    "response_time": query_duration
                }})
                
        except requests.exceptions.Timeout:
            results.append({{
                "name": test_query["name"],
                "query": test_query["query"],
                "type": test_query["type"],
                "success": False,
                "error": f"Request timed out after {{test_query['timeout']}}s",
                "response_time": test_query["timeout"]
            }})
        except Exception as e:
            results.append({{
                "name": test_query["name"],
                "query": test_query["query"],
                "type": test_query["type"],
                "success": False,
                "error": str(e)[:200],
                "response_time": time.time() - query_start if 'query_start' in locals() else 0
            }})
    
    total_duration = time.time() - start_time
    
    # Generate validation summary
    validation_summary = {{
        "success": successful_tests > 0,
        "tests_passed": successful_tests,
        "tests_total": len(test_queries),
        "results": results,
        "endpoint_name": "{endpoint_name}",
        "endpoint_url": endpoint_url,
        "duration": total_duration,
        "success_rate": successful_tests / len(test_queries) * 100,
        "endpoint_status": getattr(endpoint.state, 'ready', False) if endpoint.state else False
    }}
    
    print(json.dumps(validation_summary))
    
except Exception as outer_error:
    error_summary = {{
        "success": False,
        "tests_passed": 0,
        "tests_total": 0,
        "endpoint_name": "{endpoint_name}",
        "error": str(outer_error)[:300],
        "duration": time.time() - start_time
    }}
    print(json.dumps(error_summary))
'''
        
        result = self.executor.execute_python(
            validation_code,
            description=f"Validating endpoint: {endpoint_name}",
            timeout=timeout
        )
        
        if not result.success:
            return ValidationResult(
                success=False,
                tests_passed=0,
                tests_total=1,
                endpoint_name=endpoint_name,
                error=f"Validation execution failed: {result.error}",
                duration=result.duration
            )
        
        return ValidationResult.from_json(result.output, endpoint_name)
    
    def quick_health_check(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Perform a quick health check on the endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to check
            
        Returns:
            Health check results
        """
        logger.info(f"🏥 Quick health check: {endpoint_name}")
        
        health_check_code = f'''
import json
import time
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()

try:
    # Get endpoint status
    endpoint = workspace_client.serving_endpoints.get("{endpoint_name}")
    
    health_info = {{
        "endpoint_exists": True,
        "endpoint_name": "{endpoint_name}",
        "state_ready": getattr(endpoint.state, 'ready', False) if endpoint.state else False,
        "config_update": getattr(endpoint.state, 'config_update', None) if endpoint.state else None,
        "endpoint_url": f"{{workspace_client.config.host}}/ml/endpoints/{endpoint_name}",
        "serving_url": f"{{workspace_client.config.host}}/serving-endpoints/{endpoint_name}/invocations",
        "timestamp": time.time()
    }}
    
    print(json.dumps(health_info))
    
except Exception as e:
    error_info = {{
        "endpoint_exists": False,
        "endpoint_name": "{endpoint_name}",
        "error": str(e),
        "timestamp": time.time()
    }}
    print(json.dumps(error_info))
'''
        
        result = self.executor.execute_python(
            health_check_code,
            description=f"Health check: {endpoint_name}",
            timeout=60
        )
        
        if result.success:
            try:
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {"error": "Could not parse health check results", "output": result.output}
        else:
            return {"error": result.error, "execution_failed": True}
    
    def wait_for_endpoint_ready(self, endpoint_name: str, timeout: int = 600, 
                               check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for endpoint to become ready with periodic status checks.
        
        Args:
            endpoint_name: Name of the endpoint to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds
            
        Returns:
            Final status information
        """
        logger.info(f"⏳ Waiting for endpoint to be ready: {endpoint_name}")
        
        wait_code = f'''
import json
import time
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()
start_time = time.time()
timeout = {timeout}
check_interval = {check_interval}

status_history = []

while time.time() - start_time < timeout:
    try:
        endpoint = workspace_client.serving_endpoints.get("{endpoint_name}")
        
        current_status = {{
            "timestamp": time.time(),
            "elapsed": time.time() - start_time,
            "ready": getattr(endpoint.state, 'ready', False) if endpoint.state else False,
            "config_update": getattr(endpoint.state, 'config_update', None) if endpoint.state else None
        }}
        
        status_history.append(current_status)
        
        if current_status["ready"]:
            result = {{
                "success": True,
                "endpoint_name": "{endpoint_name}",
                "ready": True,
                "wait_time": time.time() - start_time,
                "status_checks": len(status_history),
                "final_status": current_status
            }}
            print(json.dumps(result))
            break
        
        time.sleep(check_interval)
        
    except Exception as e:
        status_history.append({{
            "timestamp": time.time(),
            "error": str(e)
        }})
        time.sleep(check_interval)

else:
    # Timeout reached
    result = {{
        "success": False,
        "endpoint_name": "{endpoint_name}",
        "ready": False,
        "timeout": True,
        "wait_time": time.time() - start_time,
        "status_checks": len(status_history),
        "error": f"Endpoint not ready after {{timeout}} seconds"
    }}
    print(json.dumps(result))
'''
        
        result = self.executor.execute_python(
            wait_code,
            description=f"Waiting for endpoint: {endpoint_name}",
            timeout=timeout + 60  # Add buffer to execution timeout
        )
        
        if result.success:
            try:
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {"error": "Could not parse wait results", "output": result.output}
        else:
            return {
                "success": False,
                "error": result.error,
                "endpoint_name": endpoint_name,
                "execution_failed": True
            }
    
    def run_load_test(self, endpoint_name: str, concurrent_requests: int = 3, 
                     requests_per_client: int = 2) -> Dict[str, Any]:
        """
        Run a simple load test on the endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to test
            concurrent_requests: Number of concurrent requests
            requests_per_client: Number of requests per concurrent client
            
        Returns:
            Load test results
        """
        logger.info(f"🚀 Running load test: {endpoint_name}")
        
        load_test_code = f'''
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()
endpoint_url = f"{{workspace_client.config.host}}/serving-endpoints/{endpoint_name}/invocations"

def make_request(client_id, request_id):
    """Make a single request to the endpoint."""
    start_time = time.time()
    try:
        payload = {{
            "input": [{{"role": "user", "content": f"Simple test query {{client_id}}-{{request_id}}: What is 2+2?"}}]
        }}
        
        response = requests.post(
            endpoint_url,
            json=payload,
            headers={{
                "Authorization": f"Bearer {{workspace_client.config.token}}",
                "Content-Type": "application/json"
            }},
            timeout=60
        )
        
        duration = time.time() - start_time
        
        return {{
            "client_id": client_id,
            "request_id": request_id,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time": duration,
            "error": None if response.status_code == 200 else response.text[:100]
        }}
        
    except Exception as e:
        return {{
            "client_id": client_id,
            "request_id": request_id,
            "success": False,
            "status_code": None,
            "response_time": time.time() - start_time,
            "error": str(e)[:100]
        }}

# Run load test
start_time = time.time()
results = []

with ThreadPoolExecutor(max_workers={concurrent_requests}) as executor:
    # Submit all requests
    futures = []
    for client_id in range({concurrent_requests}):
        for request_id in range({requests_per_client}):
            future = executor.submit(make_request, client_id, request_id)
            futures.append(future)
    
    # Collect results
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            results.append({{
                "success": False,
                "error": f"Future execution error: {{str(e)[:100]}}"
            }})

total_duration = time.time() - start_time

# Calculate statistics
successful_requests = sum(1 for r in results if r.get('success', False))
total_requests = len(results)
response_times = [r.get('response_time', 0) for r in results if r.get('success', False)]

if response_times:
    avg_response_time = sum(response_times) / len(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
else:
    avg_response_time = min_response_time = max_response_time = 0

load_test_summary = {{
    "success": successful_requests > 0,
    "endpoint_name": "{endpoint_name}",
    "total_requests": total_requests,
    "successful_requests": successful_requests,
    "failed_requests": total_requests - successful_requests,
    "success_rate": successful_requests / total_requests * 100 if total_requests > 0 else 0,
    "total_duration": total_duration,
    "requests_per_second": total_requests / total_duration if total_duration > 0 else 0,
    "avg_response_time": avg_response_time,
    "min_response_time": min_response_time,
    "max_response_time": max_response_time,
    "concurrent_clients": {concurrent_requests},
    "requests_per_client": {requests_per_client},
    "detailed_results": results
}}

print(json.dumps(load_test_summary))
'''
        
        result = self.executor.execute_python(
            load_test_code,
            description=f"Load testing: {endpoint_name}",
            timeout=300
        )
        
        if result.success:
            try:
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {"error": "Could not parse load test results", "output": result.output}
        else:
            return {
                "success": False,
                "error": result.error,
                "endpoint_name": endpoint_name,
                "execution_failed": True
            }
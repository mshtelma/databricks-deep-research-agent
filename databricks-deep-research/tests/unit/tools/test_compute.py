"""Unit tests for PythonComputeTool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import ToolDeclaration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(**kwargs: object) -> PythonComputeTool:
    return PythonComputeTool(**kwargs)  # type: ignore[arg-type]


def _ctx() -> ToolContext:
    return ToolContext(query="test")


async def _run(tool: PythonComputeTool, code: str) -> str:
    args = tool.validate_arguments({"code": code})
    result = await tool.execute(args, _ctx())
    return result.content


async def _run_result(tool: PythonComputeTool, code: str):
    args = tool.validate_arguments({"code": code})
    return await tool.execute(args, _ctx())


# ---------------------------------------------------------------------------
# Basic computation
# ---------------------------------------------------------------------------


class TestBasicComputation:
    @pytest.mark.asyncio
    async def test_arithmetic(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "2 + 2")
        assert "4" in out

    @pytest.mark.asyncio
    async def test_multiline(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "x = 10\ny = 20\nx + y")
        assert "30" in out

    @pytest.mark.asyncio
    async def test_print_capture(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "print('hello world')")
        assert "hello world" in out

    @pytest.mark.asyncio
    async def test_expression_and_print(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "print('side effect')\n42")
        assert "42" in out
        assert "side effect" in out

    @pytest.mark.asyncio
    async def test_no_output(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "x = 5")
        assert "executed successfully" in out.lower() or "no output" in out.lower()


# ---------------------------------------------------------------------------
# Module access
# ---------------------------------------------------------------------------


class TestModuleAccess:
    @pytest.mark.asyncio
    async def test_math_sqrt(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "math.sqrt(144)")
        assert "12" in out

    @pytest.mark.asyncio
    async def test_math_log(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "round(math.log(math.e), 2)")
        assert "1.0" in out

    @pytest.mark.asyncio
    async def test_statistics_mean(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "statistics.mean([1, 2, 3, 4, 5])")
        assert "3" in out

    @pytest.mark.asyncio
    async def test_statistics_geometric_mean(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "statistics.geometric_mean([2, 8])")
        assert "4" in out

    @pytest.mark.asyncio
    async def test_statistics_stdev(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "round(statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]), 2)")
        assert "2" in out


# ---------------------------------------------------------------------------
# Variable persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_variable_persists_across_calls(self) -> None:
        tool = _make_tool()
        await _run(tool, "data = [100, 200, 300]")
        out = await _run(tool, "sum(data)")
        assert "600" in out

    @pytest.mark.asyncio
    async def test_function_persists(self) -> None:
        tool = _make_tool()
        await _run(tool, "def percent_change(a, b): return (b - a) / a * 100")
        out = await _run(tool, "percent_change(100, 150)")
        assert "50" in out

    @pytest.mark.asyncio
    async def test_separate_instances_isolated(self) -> None:
        tool1 = _make_tool()
        tool2 = _make_tool()
        await _run(tool1, "secret = 42")
        result = await _run_result(tool2, "secret")
        assert result.success is False or "NameError" in result.content


# ---------------------------------------------------------------------------
# Security: blocked operations
# ---------------------------------------------------------------------------


class TestSandboxSecurity:
    @pytest.mark.asyncio
    async def test_import_os_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import os")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_from_import_os_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "from os import path")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_import_math_allowed(self) -> None:
        """LLMs sometimes write 'import math' despite prompt — should work."""
        tool = _make_tool()
        result = await _run_result(tool, "import math\nprint(math.sqrt(16))")
        assert result.success
        assert "4.0" in result.content

    @pytest.mark.asyncio
    async def test_from_statistics_import_allowed(self) -> None:
        """from statistics import mean should work via _restricted_import."""
        tool = _make_tool()
        result = await _run_result(tool, "from statistics import mean\nprint(mean([1,2,3]))")
        assert result.success
        assert "2" in result.content

    @pytest.mark.asyncio
    async def test_import_numpy_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import numpy")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_open_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "open('/etc/passwd')")
        assert result.success is False
        assert "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_eval_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "eval('1+1')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_exec_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "exec('import os')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_compile_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "compile('1+1', '<x>', 'eval')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_getattr_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "getattr(math, '__builtins__')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_dunder_import_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "__import__('os')")
        assert result.success is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_syntax_error(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "def ()")
        assert result.success is False
        assert "SyntaxError" in result.content

    @pytest.mark.asyncio
    async def test_name_error(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "undefined_variable")
        assert result.success is False
        assert "NameError" in result.content

    @pytest.mark.asyncio
    async def test_division_by_zero(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "1 / 0")
        assert result.success is False
        assert "ZeroDivision" in result.content

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        tool = _make_tool(max_execution_seconds=0.5)
        # Use a CPU-bound loop that will time out. The orphaned thread will
        # eventually be cleaned up when the process exits.
        result = await _run_result(tool, "sum(range(10**9))")
        assert result.success is False
        assert "timeout" in result.content.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_code_rejected(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="non-empty"):
            tool.validate_arguments({"code": ""})

    def test_whitespace_only_rejected(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="non-empty"):
            tool.validate_arguments({"code": "   "})

    def test_code_too_long_rejected(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="maximum length"):
            tool.validate_arguments({"code": "x" * 6000})

    def test_valid_code_passes(self) -> None:
        tool = _make_tool()
        result = tool.validate_arguments({"code": "  2 + 2  "})
        assert result["code"] == "2 + 2"


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestOutputTruncation:
    @pytest.mark.asyncio
    async def test_large_output_truncated(self) -> None:
        tool = _make_tool(max_output_chars=100)
        out = await _run(tool, "print('x' * 500)")
        assert len(out) <= 200  # truncated + "... (output truncated)" suffix
        assert "truncated" in out.lower()


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_definition_name(self) -> None:
        tool = _make_tool(name="my_compute")
        assert tool.definition.name == "my_compute"

    def test_definition_source_kind(self) -> None:
        tool = _make_tool()
        assert tool.definition.source_kind == "builtin"

    def test_definition_has_code_parameter(self) -> None:
        tool = _make_tool()
        params = tool.definition.parameters
        assert "code" in params["properties"]
        assert "code" in params["required"]


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    @pytest.mark.asyncio
    async def test_factory_creates_compute_tool(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="compute", kind="compute", config={})
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "compute"
        assert isinstance(tool, PythonComputeTool)

    @pytest.mark.asyncio
    async def test_factory_passes_config(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="calc",
            kind="compute",
            config={"max_execution_seconds": 30.0, "max_output_chars": 5000},
            description="Custom calculator",
        )
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)

        assert tool._max_execution_seconds == 30.0  # type: ignore[attr-defined]
        assert tool._max_output_chars == 5000  # type: ignore[attr-defined]

    def test_factory_supports_compute(self) -> None:
        factory = BuiltinToolFactory()
        assert factory.supports("compute")


# ---------------------------------------------------------------------------
# Extended module whitelist (Fix 8)
# ---------------------------------------------------------------------------


class TestExtendedModuleWhitelist:
    @pytest.mark.asyncio
    async def test_import_decimal_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import decimal\nprint(decimal.Decimal('1.1') + decimal.Decimal('2.2'))")
        assert result.success
        assert "3.3" in result.content

    @pytest.mark.asyncio
    async def test_import_re_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import re\nprint(len(re.findall(r'\\d+', 'abc 123 def 456')))")
        assert result.success
        assert "2" in result.content

    @pytest.mark.asyncio
    async def test_import_fractions_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import fractions\nprint(fractions.Fraction(1, 3) + fractions.Fraction(1, 6))")
        assert result.success
        assert "1/2" in result.content

    @pytest.mark.asyncio
    async def test_import_itertools_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import itertools\nprint(list(itertools.combinations([1,2,3], 2)))")
        assert result.success
        assert "(1, 2)" in result.content

    @pytest.mark.asyncio
    async def test_import_functools_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import functools\nprint(functools.reduce(lambda a, b: a + b, [1,2,3,4]))")
        assert result.success
        assert "10" in result.content

    @pytest.mark.asyncio
    async def test_import_collections_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import collections\nprint(collections.Counter('abracadabra').most_common(2))")
        assert result.success
        assert "'a'" in result.content

    @pytest.mark.asyncio
    async def test_import_subprocess_still_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import subprocess")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_import_sys_still_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import sys")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_decimal_persists_across_calls(self) -> None:
        """After import decimal, module should persist in namespace."""
        tool = _make_tool()
        await _run(tool, "import decimal\nx = decimal.Decimal('3.14')")
        out = await _run(tool, "print(x * 2)")
        assert "6.28" in out

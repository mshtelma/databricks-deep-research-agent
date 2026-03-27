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
            tool.validate_arguments({"code": "x" * 21000})

    def test_custom_code_length_limit(self) -> None:
        from databricks_deep_research.tools.builtins.compute import PythonComputeTool
        tool = PythonComputeTool(max_code_length=5000)
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


# ---------------------------------------------------------------------------
# Security: sandbox escape prevention (AST dunder blocker)
# ---------------------------------------------------------------------------


class TestSandboxEscapePrevention:
    @pytest.mark.asyncio
    async def test_dunder_builtins_access_blocked(self) -> None:
        """statistics.__builtins__ must not be accessible."""
        tool = _make_tool()
        result = await _run_result(tool, "statistics.__builtins__")
        assert result.success is False
        assert "__builtins__" in result.content

    @pytest.mark.asyncio
    async def test_dunder_class_access_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "().__class__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_class_hierarchy_chain_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "().__class__.__bases__[0].__subclasses__()")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_module_globals_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "math.sqrt.__globals__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_dunder_dict_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "math.__dict__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_safe_dunders_allowed(self) -> None:
        """Legitimate dunders like __len__ should still work."""
        tool = _make_tool()
        result = await _run_result(tool, "[1,2,3].__len__()")
        assert result.success
        assert "3" in result.content

    @pytest.mark.asyncio
    async def test_safe_dunder_add_allowed(self) -> None:
        """Arithmetic dunders should work."""
        tool = _make_tool()
        result = await _run_result(tool, "(5).__add__(3)")
        assert result.success
        assert "8" in result.content

    @pytest.mark.asyncio
    async def test_safe_dunder_contains_allowed(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "'hello'.__contains__('ell')")
        assert result.success
        assert "True" in result.content


# ---------------------------------------------------------------------------
# Per-instance module isolation
# ---------------------------------------------------------------------------


class TestPerInstanceModules:
    @pytest.mark.asyncio
    async def test_allowed_modules_restricts(self) -> None:
        """allowed_modules should restrict to only listed modules."""
        tool = _make_tool(allowed_modules=["math"])
        result = await _run_result(tool, "import statistics")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_allowed_modules_includes_requested(self) -> None:
        tool = _make_tool(allowed_modules=["math", "json"])
        result = await _run_result(tool, "import json\nprint(json.dumps({'a': 1}))")
        assert result.success
        assert '"a"' in result.content

    @pytest.mark.asyncio
    async def test_two_instances_independent(self) -> None:
        """Two instances with different allowed_modules must not leak."""
        tool_a = _make_tool(allowed_modules=["math", "json"])
        tool_b = _make_tool(allowed_modules=["math"])
        # A can import json
        result_a = await _run_result(tool_a, "import json\njson.dumps({})")
        assert result_a.success
        # B cannot
        result_b = await _run_result(tool_b, "import json")
        assert result_b.success is False

    @pytest.mark.asyncio
    async def test_extra_modules_extends_defaults(self) -> None:
        """extra_modules should extend (not replace) defaults."""
        # Use a stdlib module that's NOT in the default list as extra
        tool = _make_tool(extra_modules=["uuid"])
        result = await _run_result(tool, "import uuid\nprint(type(uuid.uuid4()))")
        assert result.success
        assert "UUID" in result.content
        # Default modules still available
        result2 = await _run_result(tool, "import math\nmath.sqrt(4)")
        assert result2.success

    @pytest.mark.asyncio
    async def test_extra_module_not_installed_skipped(self) -> None:
        """Non-existent extra module is skipped gracefully."""
        tool = _make_tool(extra_modules=["nonexistent_module_xyz_123"])
        result = await _run_result(tool, "math.sqrt(4)")
        assert result.success
        assert "2" in result.content


# ---------------------------------------------------------------------------
# New stdlib modules
# ---------------------------------------------------------------------------


class TestNewStdlibModules:
    @pytest.mark.asyncio
    async def test_datetime_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import datetime\nprint(datetime.date(2024, 1, 15))"
        )
        assert result.success
        assert "2024-01-15" in result.content

    @pytest.mark.asyncio
    async def test_json_loads_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import json\nprint(json.loads('{\"a\": 1}'))"
        )
        assert result.success
        assert "'a'" in result.content

    @pytest.mark.asyncio
    async def test_operator_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import operator\nprint(operator.add(2, 3))"
        )
        assert result.success
        assert "5" in result.content

    @pytest.mark.asyncio
    async def test_string_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import string\nprint(string.ascii_lowercase[:5])"
        )
        assert result.success
        assert "abcde" in result.content

    @pytest.mark.asyncio
    async def test_textwrap_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import textwrap\nprint(textwrap.shorten('Hello World!', width=8))"
        )
        assert result.success
        assert "[...]" in result.content or "Hello" in result.content


# ---------------------------------------------------------------------------
# Namespace safety
# ---------------------------------------------------------------------------


class TestNamespaceSafety:
    @pytest.mark.asyncio
    async def test_variable_cannot_shadow_module(self) -> None:
        """Assigning math = 42 must not kill the math module on next call."""
        tool = _make_tool()
        await _run(tool, "math = 42")
        result = await _run_result(tool, "math.sqrt(9)")
        assert result.success
        assert "3" in result.content

    @pytest.mark.asyncio
    async def test_code_length_20k_accepted(self) -> None:
        """Code up to 20K chars should be accepted."""
        tool = _make_tool()
        code = "x = 1\n" * 3300  # ~19.8K chars
        code += "x"
        result = tool.validate_arguments({"code": code})
        assert "code" in result

    @pytest.mark.asyncio
    async def test_code_length_over_limit_rejected(self) -> None:
        """Code over the limit should be rejected."""
        tool = _make_tool()
        with pytest.raises(ValueError, match="maximum length"):
            tool.validate_arguments({"code": "x" * 21000})


# ---------------------------------------------------------------------------
# Sandbox introspection contract (Change 6e)
# ---------------------------------------------------------------------------


class TestSandboxIntrospectionContract:
    """Verify sandbox contract: introspection builtins are not available,
    but try/except NameError works for variable existence checks."""

    @pytest.mark.asyncio
    async def test_globals_not_available(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "globals()")
        assert result.success is False
        assert "NameError" in result.content

    @pytest.mark.asyncio
    async def test_vars_not_available(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "vars()")
        assert result.success is False
        assert "NameError" in result.content

    @pytest.mark.asyncio
    async def test_locals_not_available(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "locals()")
        assert result.success is False
        assert "NameError" in result.content

    @pytest.mark.asyncio
    async def test_dir_not_available(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "dir()")
        assert result.success is False
        assert "NameError" in result.content

    @pytest.mark.asyncio
    async def test_try_bare_except_works(self) -> None:
        """Bare except works for variable existence checks."""
        tool = _make_tool()
        out = await _run(tool, "try:\n    missing_var\nexcept:\n    print('not found')")
        assert "not found" in out

    @pytest.mark.asyncio
    async def test_try_except_with_existing_var(self) -> None:
        tool = _make_tool()
        await _run(tool, "op1 = 42")
        out = await _run(tool, "try:\n    print('op1 =', op1)\nexcept:\n    print('not found')")
        assert "op1 = 42" in out

    @pytest.mark.asyncio
    async def test_cross_call_dict_growth(self) -> None:
        """Dict created in call 1, mutated in call 2, persists."""
        tool = _make_tool()
        await _run(tool, "d = {}; d['op1'] = 100")
        await _run(tool, "d['op2'] = 200")
        out = await _run(tool, "print(d)")
        assert "'op1': 100" in out
        assert "'op2': 200" in out

    # -- namespace_snapshot() tests (Change 7) --------------------------------

    @pytest.mark.asyncio
    async def test_namespace_snapshot_returns_stored_vars(self) -> None:
        tool = _make_tool()
        await _run(tool, "op1 = 42")
        await _run(tool, "op2 = 99")
        snap = tool.namespace_snapshot()
        assert "op1 = 42" in snap
        assert "op2 = 99" in snap

    def test_namespace_snapshot_empty(self) -> None:
        tool = _make_tool()
        snap = tool.namespace_snapshot()
        assert snap == "(empty — no variables stored)"

    @pytest.mark.asyncio
    async def test_namespace_snapshot_filters_underscore_prefix(self) -> None:
        tool = _make_tool()
        await _run(tool, "_tmp = 1")
        await _run(tool, "op1 = 2")
        snap = tool.namespace_snapshot()
        assert "op1 = 2" in snap
        assert "_tmp" not in snap

    @pytest.mark.asyncio
    async def test_namespace_snapshot_truncates_long_values(self) -> None:
        tool = _make_tool()
        await _run(tool, "big = list(range(100))")
        snap = tool.namespace_snapshot()
        assert "big = " in snap
        assert "..." in snap

    @pytest.mark.asyncio
    async def test_namespace_snapshot_total_truncation(self) -> None:
        """Store many variables — total capped at 2000 chars with '... (N more)'."""
        tool = _make_tool()
        # Create 50 variables with moderately long values
        for i in range(50):
            await _run(tool, f"var_{i:03d} = 'value_{i:03d}_padding_text_here'")
        snap = tool.namespace_snapshot()
        assert len(snap) <= 2200  # allow small overshoot for the truncation message
        assert "... (" in snap
        assert "more variables)" in snap

    @pytest.mark.asyncio
    async def test_namespace_snapshot_excludes_non_safe_types(self) -> None:
        """Non-safe-type objects (custom classes) are excluded from snapshot."""
        tool = _make_tool()
        await _run(tool, "op1 = 42")
        # Inject a non-safe-type object directly
        tool._namespace["broken"] = type("Broken", (), {})()
        snap = tool.namespace_snapshot()
        assert "op1 = 42" in snap
        assert "broken" not in snap

    @pytest.mark.asyncio
    async def test_namespace_snapshot_contains_dict_with_braces(self) -> None:
        """Dict values with braces appear literally (no escaping needed)."""
        tool = _make_tool()
        await _run(tool, "d = {'a': 1, 'b': 2}")
        snap = tool.namespace_snapshot()
        assert "d = " in snap
        assert "'a'" in snap
        assert "'b'" in snap


# ---------------------------------------------------------------------------
# Submodule imports (Change 8 — Phase 1a)
# ---------------------------------------------------------------------------


class TestSubmoduleImports:
    @pytest.mark.asyncio
    async def test_submodule_from_import(self) -> None:
        """from collections.abc import Mapping should work."""
        tool = _make_tool()
        result = await _run_result(
            tool, "from collections.abc import Mapping\nprint(isinstance({}, Mapping))"
        )
        assert result.success
        assert "True" in result.content

    @pytest.mark.asyncio
    async def test_submodule_bare_import(self) -> None:
        """import collections.abc — returns root module per CPython protocol."""
        tool = _make_tool()
        result = await _run_result(
            tool, "import collections\nfrom collections import OrderedDict\nprint(OrderedDict())"
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_submodule_nonexistent_raises(self) -> None:
        """import collections.nonexistent → clear ImportError."""
        tool = _make_tool()
        result = await _run_result(tool, "from collections import nonexistent_xyz")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_blocked_root_with_submodule(self) -> None:
        """import os.path → ImportError (os not whitelisted)."""
        tool = _make_tool()
        result = await _run_result(tool, "import os.path")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_submodule_extra_numpy(self) -> None:
        """from numpy.linalg import norm — via extra_modules."""
        pytest.importorskip("numpy")
        tool = _make_tool(extra_modules=["numpy"])
        result = await _run_result(
            tool, "from numpy.linalg import norm\nprint(norm([3, 4]))"
        )
        assert result.success
        assert "5" in result.content


# ---------------------------------------------------------------------------
# Safe builtins expansion (Change 8 — Phase 1b)
# ---------------------------------------------------------------------------


class TestSafeBuiltinsExpansion:
    @pytest.mark.asyncio
    async def test_exception_types_in_try_except(self) -> None:
        """try: 1/0 except ZeroDivisionError: 'caught'."""
        tool = _make_tool()
        out = await _run(
            tool, "try:\n    1/0\nexcept ZeroDivisionError:\n    print('caught')"
        )
        assert "caught" in out

    @pytest.mark.asyncio
    async def test_hasattr_available(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "hasattr([], 'append')")
        assert "True" in out

    @pytest.mark.asyncio
    async def test_hex_bin_oct(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "print(hex(255), bin(10), oct(8))")
        assert "0xff" in out
        assert "0b1010" in out
        assert "0o10" in out

    @pytest.mark.asyncio
    async def test_value_error_catchable(self) -> None:
        tool = _make_tool()
        out = await _run(
            tool, "try:\n    int('abc')\nexcept ValueError:\n    print('caught ValueError')"
        )
        assert "caught ValueError" in out

    @pytest.mark.asyncio
    async def test_key_error_catchable(self) -> None:
        tool = _make_tool()
        out = await _run(
            tool, "try:\n    {}['missing']\nexcept KeyError:\n    print('caught KeyError')"
        )
        assert "caught KeyError" in out

    @pytest.mark.asyncio
    async def test_name_error_catchable(self) -> None:
        tool = _make_tool()
        out = await _run(
            tool, "try:\n    undefined_xyz\nexcept NameError:\n    print('caught NameError')"
        )
        assert "caught NameError" in out


# ---------------------------------------------------------------------------
# list_user_namespace() (Change 8 — Phase 1c)
# ---------------------------------------------------------------------------


class TestListUserNamespace:
    @pytest.mark.asyncio
    async def test_list_user_namespace_basic(self) -> None:
        tool = _make_tool()
        await _run(tool, "op1 = 42")
        await _run(tool, "op2 = 99")
        entries = tool.list_user_namespace()
        names = {e["name"] for e in entries}
        assert "op1" in names
        assert "op2" in names

    @pytest.mark.asyncio
    async def test_list_user_namespace_prefix(self) -> None:
        tool = _make_tool()
        await _run(tool, "op1 = 10")
        await _run(tool, "op2 = 20")
        await _run(tool, "total = 30")
        entries = tool.list_user_namespace(prefix="op")
        names = {e["name"] for e in entries}
        assert names == {"op1", "op2"}

    @pytest.mark.asyncio
    async def test_list_user_namespace_names(self) -> None:
        tool = _make_tool()
        await _run(tool, "op1 = 10")
        await _run(tool, "op2 = 20")
        await _run(tool, "op3 = 30")
        entries = tool.list_user_namespace(names=["op1"])
        assert len(entries) == 1
        assert entries[0]["name"] == "op1"

    @pytest.mark.asyncio
    async def test_list_user_namespace_filters_modules(self) -> None:
        """Stored module refs are excluded (not safe type)."""
        tool = _make_tool()
        await _run(tool, "import math")
        # math module in namespace should be filtered out
        entries = tool.list_user_namespace()
        names = {e["name"] for e in entries}
        assert "math" not in names

    @pytest.mark.asyncio
    async def test_list_user_namespace_filters_underscore(self) -> None:
        tool = _make_tool()
        await _run(tool, "_tmp = 1")
        await _run(tool, "visible = 2")
        entries = tool.list_user_namespace()
        names = {e["name"] for e in entries}
        assert "_tmp" not in names
        assert "visible" in names

    @pytest.mark.asyncio
    async def test_list_user_namespace_truncates(self) -> None:
        tool = _make_tool()
        await _run(tool, "big = list(range(200))")
        entries = tool.list_user_namespace(max_value_repr=50)
        assert len(entries) == 1
        assert entries[0]["value"].endswith("...")

"""Unit tests for PythonComputeTool."""

from __future__ import annotations

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
    async def test_getattribute_string_reflection_blocked(self) -> None:
        """``__getattribute__`` smuggles a blocked attr NAME as a STRING past the
        AST attr-name guard (security review R4). It must be denylisted so the
        ``type(object).__getattribute__(object, "__subclasses__")() ->
        _wrap_close.__init__.__globals__["system"]`` RCE escape stays closed."""
        tool = _make_tool()
        result = await _run_result(
            tool, 'type(object).__getattribute__(object, "__subclasses__")'
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_getattribute_method_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, '().__getattribute__("__class__")')
        assert result.success is False

    @pytest.mark.asyncio
    async def test_base_and_subclasshook_blocked(self) -> None:
        tool = _make_tool()
        assert (await _run_result(tool, "int.__base__")).success is False
        assert (await _run_result(tool, "int.__subclasshook__")).success is False

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

    # -- R5: string-keyed attribute-fetcher escape class -------------------
    # ``operator`` and ``string`` exposed runtime string->attribute fetchers
    # (``attrgetter``/``methodcaller``/``Formatter().get_field``/``vformat``)
    # INVISIBLE to the AST attr-name guard. They are removed from the sandbox,
    # so every escape vector dies on the import line (ImportError).

    @pytest.mark.asyncio
    async def test_operator_import_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import operator")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_operator_from_import_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "from operator import attrgetter")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_string_import_blocked(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import string")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_string_formatter_unreachable(self) -> None:
        """``string.Formatter`` must be unreachable — its ``get_field``/``vformat``
        resolve a dotted attribute path from a STRING (RCE vector P2)."""
        tool = _make_tool()
        result = await _run_result(tool, "from string import Formatter")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_r5_p1_attrgetter_class_blocked(self) -> None:
        """R5 P1: ``operator.attrgetter('__class__')`` fetches a denylisted attr
        by string. Dies on the import line."""
        tool = _make_tool()
        result = await _run_result(
            tool, "import operator\noperator.attrgetter('__class__')(())"
        )
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_r5_p2_formatter_get_field_blocked(self) -> None:
        """R5 P2: ``string.Formatter().get_field('w.__init__.__globals__', ...)``
        resolves a dotted attr path from a string. Dies on the import line."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "import string\n"
            "string.Formatter().get_field('w.__init__.__globals__', [], {'w': ()})",
        )
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_r5_p4_methodcaller_subclasses_blocked(self) -> None:
        """R5 P4: ``operator.methodcaller('__subclasses__')(object)`` invokes a
        denylisted method by string. Dies on the import line."""
        tool = _make_tool()
        result = await _run_result(
            tool, "import operator\noperator.methodcaller('__subclasses__')(object)"
        )
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_str_format_field_access_cannot_reach_globals(self) -> None:
        """HONEST residual-risk assertion (review R5).

        ``str.format`` field-access (``"{0.__class__}".format(obj)``) is a str
        METHOD — it cannot be removed and IS invisible to the AST attr-name guard
        (the attrs live inside a str literal). It CAN read a bare class object,
        but the format mini-language cannot express a CALL ``()``, so it cannot
        reach ``__subclasses__()`` (a call) nor ``__globals__`` off a slot-wrapper
        — i.e. it cannot escalate to RCE. This test pins that boundary: the
        chained ``"{0.__init__.__globals__}".format(())`` does NOT yield a usable
        globals dict.
        """
        tool = _make_tool()
        # The RCE-escalation chain fails inside Python itself (no usable globals).
        result = await _run_result(
            tool, '"{0.__init__.__globals__}".format(())'
        )
        assert result.success is False
        assert "__globals__" in result.content
        # And a CALL cannot be expressed in the format mini-language: the
        # ``__subclasses__()`` text is treated as a literal attribute name.
        result_call = await _run_result(
            tool, '"{0.__class__.__subclasses__()}".format(())'
        )
        assert result_call.success is False
        # Documented residual: a bare class object CAN be read (no escalation).
        result_class = await _run_result(tool, '"{0.__class__}".format(())')
        assert result_class.success is True
        assert "tuple" in result_class.content


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
    async def test_extra_modules_arbitrary_stdlib_rejected(self) -> None:
        """extra_modules outside the vetted allowlist is REJECTED at construction
        (security review R7 — MEDIUM). ``uuid`` is a pure stdlib module but is NOT
        in the allowlist (it re-exports ``os``/``sys`` as public attrs), so the
        old "extend with any stdlib module" behaviour is now closed."""
        with pytest.raises(ValueError, match="not permitted"):
            _make_tool(extra_modules=["uuid"])

    @pytest.mark.asyncio
    async def test_extra_module_not_in_allowlist_rejected(self) -> None:
        """A module name absent from the vetted allowlist is rejected at
        construction (not silently skipped) — a non-existent name is also not in
        the allowlist, so it is rejected rather than degrading to a warning."""
        with pytest.raises(ValueError, match="not permitted"):
            _make_tool(extra_modules=["nonexistent_module_xyz_123"])


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
    async def test_operator_module_removed(self) -> None:
        """``operator`` was removed from the sandbox (review R5): it exposed
        ``attrgetter``/``methodcaller`` — string-keyed attribute fetchers that
        bypass the AST attr-name guard and are proven RCE."""
        tool = _make_tool()
        result = await _run_result(
            tool, "import operator\nprint(operator.add(2, 3))"
        )
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_string_module_removed(self) -> None:
        """``string`` was removed from the sandbox (review R5): it exposed
        ``Formatter().get_field``/``vformat`` — string-keyed attribute fetchers
        that bypass the AST attr-name guard and are proven RCE."""
        tool = _make_tool()
        result = await _run_result(
            tool, "import string\nprint(string.ascii_lowercase[:5])"
        )
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_textwrap_works(self) -> None:
        tool = _make_tool()
        result = await _run_result(
            tool, "import textwrap\nprint(textwrap.shorten('Hello World!', width=8))"
        )
        assert result.success
        assert "[...]" in result.content or "Hello" in result.content


# ---------------------------------------------------------------------------
# Legitimate compute still works after the R5 operator/string removal
# ---------------------------------------------------------------------------


class TestLegitimateComputeAfterR5Removal:
    """The R5 fix removes ONLY ``operator``/``string``; every other module and
    a representative data-analysis snippet must still run."""

    @pytest.mark.asyncio
    async def test_remaining_modules_still_importable(self) -> None:
        tool = _make_tool()
        for mod in (
            "math",
            "statistics",
            "decimal",
            "re",
            "fractions",
            "itertools",
            "functools",
            "collections",
            "copy",
            "calendar",
            "datetime",
            "json",
            "textwrap",
        ):
            result = await _run_result(tool, f"import {mod}")
            assert result.success, f"{mod} should still import: {result.content}"

    @pytest.mark.asyncio
    async def test_representative_data_analysis_snippet(self) -> None:
        """A realistic multi-module compute task runs end-to-end."""
        tool = _make_tool()
        code = (
            "import statistics, math, functools, itertools, collections, json\n"
            "from collections import Counter\n"
            "rows = [{'cat': 'a', 'v': 10}, {'cat': 'b', 'v': 20},\n"
            "        {'cat': 'a', 'v': 30}, {'cat': 'b', 'v': 40}]\n"
            "vals = [r['v'] for r in rows]\n"
            "total = functools.reduce(lambda x, y: x + y, vals)\n"
            "by_cat = collections.defaultdict(list)\n"
            "for r in rows:\n"
            "    by_cat[r['cat']].append(r['v'])\n"
            "means = {k: statistics.mean(v) for k, v in by_cat.items()}\n"
            "counts = Counter(r['cat'] for r in rows)\n"
            "stdev = round(statistics.pstdev(vals), 4)\n"
            "rms = round(math.sqrt(statistics.mean([v * v for v in vals])), 4)\n"
            "pairs = list(itertools.combinations(sorted(set(vals)), 2))\n"
            "print(json.dumps({'total': total, 'means': means,\n"
            "                  'counts': dict(counts), 'stdev': stdev,\n"
            "                  'rms': rms, 'n_pairs': len(pairs)}))"
        )
        result = await _run_result(tool, code)
        assert result.success, result.content
        assert '"total": 100' in result.content
        assert '"a": 20' in result.content  # mean of [10, 30]
        assert '"b": 30' in result.content  # mean of [20, 40]

    @pytest.mark.asyncio
    async def test_functools_partial_and_reduce_still_work(self) -> None:
        """``functools.partial``/``reduce`` are pure (not string-keyed fetchers)
        and remain available."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "import functools\n"
            "add = functools.partial(lambda a, b: a + b, 10)\n"
            "print(add(5), functools.reduce(lambda x, y: x * y, [1, 2, 3, 4]))",
        )
        assert result.success, result.content
        assert "15" in result.content  # partial(10) + 5
        assert "24" in result.content  # 1*2*3*4

    @pytest.mark.asyncio
    async def test_re_compile_is_regex_not_code(self) -> None:
        """``re.compile`` compiles a REGEX (safe), not Python code."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "import re\n"
            "print(re.compile(r'\\d+').findall('a1b22c333'))",
        )
        assert result.success, result.content
        assert "'1'" in result.content and "'333'" in result.content

    @pytest.mark.asyncio
    async def test_default_allowed_modules_set_is_exactly_expected(self) -> None:
        """Pin the default module allowlist: operator/string are gone, nothing
        else changed (the default path is byte-identical except the removals)."""
        from databricks_deep_research.tools.builtins.compute import (
            _ALLOWED_IMPORT_MODULES,
        )

        assert "operator" not in _ALLOWED_IMPORT_MODULES
        assert "string" not in _ALLOWED_IMPORT_MODULES
        assert set(_ALLOWED_IMPORT_MODULES) == {
            "math",
            "statistics",
            "decimal",
            "re",
            "fractions",
            "itertools",
            "functools",
            "collections",
            "copy",
            "calendar",
            "datetime",
            "json",
            "textwrap",
        }


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
        """numpy via ``extra_modules`` is now routed through the SAFE dataframe
        facade (security review R7), so ``from numpy.linalg import norm`` is
        rejected (R2 escape fix) — the supported form is ``import numpy as np``
        + the curated ``np.linalg.norm`` sub-facade."""
        pytest.importorskip("numpy")
        tool = _make_tool(extra_modules=["numpy"])
        # The legacy from-import form is now rejected (facaded-root hardening).
        rejected = await _run_result(
            tool, "from numpy.linalg import norm\nprint(norm([3, 4]))"
        )
        assert rejected.success is False
        assert "not allowed" in rejected.content.lower()
        # The safe-path equivalent works via the curated sub-facade.
        result = await _run_result(
            tool, "import numpy as np\nprint(np.linalg.norm([3, 4]))"
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


# ---------------------------------------------------------------------------
# Security: R6 live-module attribute-reach escape class (generic facade)
# ---------------------------------------------------------------------------
#
# Review R6 proved RCE in DEFAULT mode (no import statement) via REAL stdlib
# module objects pre-injected into the sandbox that re-export ``sys``/``os``/
# ``operator`` as PLAIN non-dunder attributes:
#   calendar.sys.modules['os'].system('id')        # calendar does ``import sys``
#   fractions.operator.methodcaller('__subclasses__')(type(()))
#   fractions.sys / statistics.sys / collections._sys / re.functools / ...
# The structural fix wraps EVERY allowed module in a curated facade that drops
# every module-typed attribute, so NO live module object is reachable.


# The complete default-mode allowed module set (mirrors _ALLOWED_IMPORT_MODULES).
_ALL_FACETED_MODULES: tuple[str, ...] = (
    "math",
    "statistics",
    "decimal",
    "re",
    "fractions",
    "itertools",
    "functools",
    "collections",
    "copy",
    "calendar",
    "datetime",
    "json",
    "textwrap",
)


class TestR6LiveModuleAttributeReachBlocked:
    """Every R6 attribute-reach escape (a re-exported real module on a pre-injected
    module) must die with AttributeError — the facade simply has no such attr."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expr",
        [
            # The headline R6 escapes (proven RCE).
            "calendar.sys",
            "fractions.operator",
            "fractions.sys",
            "statistics.sys",
            "collections._sys",
            "re.functools",
            # Every other re-exported live module on an allowed module.
            "statistics.math",
            "statistics.numbers",
            "statistics.random",
            "fractions.math",
            "fractions.numbers",
            "fractions.re",
            "re.enum",
            "re.copyreg",
            "calendar.datetime",
            "datetime.sys",
            "json.decoder",
            "json.encoder",
            "json.scanner",
            "json.codecs",
            "textwrap.re",
            "collections.abc",
        ],
    )
    async def test_module_typed_attr_unreachable(self, expr: str) -> None:
        tool = _make_tool()
        result = await _run_result(tool, expr)
        assert result.success is False, expr
        assert "AttributeError" in result.content or "not allowed" in result.content

    @pytest.mark.asyncio
    async def test_r6_calendar_sys_modules_os_system_full_chain_blocked(self) -> None:
        """The full proven R6 RCE chain dies at the first ``calendar.sys`` reach."""
        tool = _make_tool()
        result = await _run_result(
            tool, "calendar.sys.modules['os'].system('id')"
        )
        assert result.success is False
        assert "AttributeError" in result.content

    @pytest.mark.asyncio
    async def test_r6_fractions_operator_methodcaller_subclasses_blocked(self) -> None:
        """``fractions.operator.methodcaller('__subclasses__')(type(()))`` —
        resurrecting the R5-removed ``operator`` via ``fractions`` — is blocked at
        the ``fractions.operator`` reach (faceted away)."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "fractions.operator.methodcaller('__subclasses__')(type(()))",
        )
        assert result.success is False
        assert "AttributeError" in result.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("module_name", _ALL_FACETED_MODULES)
    async def test_no_module_typed_attr_via_dir(self, module_name: str) -> None:
        """STRUCTURAL invariant: for EVERY allowed module, ``dir()`` exposes ZERO
        ModuleType attributes (the facade dropped every module handle)."""
        tool = _make_tool()
        facade = tool._allowed_modules[module_name]  # type: ignore[attr-defined]
        leaked = [
            name
            for name in dir(facade)
            if not name.startswith("_")
            and isinstance(getattr(facade, name, None), __import__("types").ModuleType)
        ]
        assert leaked == [], f"{module_name} leaks module-typed attrs: {leaked}"

    def test_injected_namespace_is_entirely_facades(self) -> None:
        """Every module the sandbox can see is a FACADE, never a real stdlib
        module (build-time structural guarantee, asserted via identity)."""
        import math as real_math
        import statistics as real_statistics

        tool = _make_tool()
        # The injected ``math`` must NOT be the real math module.
        assert tool._modules["math"] is not real_math  # type: ignore[attr-defined]
        assert tool._modules["statistics"] is not real_statistics  # type: ignore[attr-defined]
        # But the legitimate callable is preserved (same function object).
        assert tool._modules["math"].sqrt is real_math.sqrt  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_from_root_import_live_submodule_blocked(self) -> None:
        """``from collections import abc`` (``abc`` is a live submodule) must NOT
        bind a live module via CPython's IMPORT_FROM ``sys.modules`` fallback."""
        tool = _make_tool()
        result = await _run_result(tool, "from collections import abc")
        assert result.success is False
        assert "cannot import name" in result.content
        # And it did not leak the module into the persistent namespace.
        assert tool.get_variable("abc") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            "from re import enum",
            "from re import functools",
            "from statistics import math",
            "from statistics import sys",
            "from fractions import operator",
            "from fractions import sys",
            "from calendar import sys",
            "from datetime import sys",
            "from json import decoder",
        ],
    )
    async def test_from_root_import_module_typed_name_blocked(self, code: str) -> None:
        """``from <module> import <a-submodule/re-exported-module>`` is rejected —
        the facade lacks the name, so the dangerous ``sys.modules`` fallback is
        pre-empted with a clean ImportError."""
        tool = _make_tool()
        result = await _run_result(tool, code)
        assert result.success is False, code
        assert "cannot import name" in result.content

    @pytest.mark.asyncio
    async def test_live_module_never_persists_across_calls(self) -> None:
        """Defense-in-depth: even if a binding path produced a module, it must not
        persist. ``from collections.abc import Mapping`` binds only ``Mapping``."""
        tool = _make_tool()
        await _run(tool, "from collections.abc import Mapping")
        # Mapping (a class) persists; no module handle does.
        import types as _types

        for _name, value in tool._namespace.items():  # type: ignore[attr-defined]
            assert not isinstance(value, _types.ModuleType)


class TestR6PublicApiPreserved:
    """Each faceted module keeps its full legitimate public API after R6."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code,expected",
        [
            # math
            ("math.sqrt(144)", "12"),
            ("round(math.log(math.e), 2)", "1.0"),
            # statistics
            ("statistics.mean([1, 2, 3, 4, 5])", "3"),
            ("statistics.median([1, 3, 5])", "3"),
            ("round(statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]), 2)", "2.14"),
            ("statistics.geometric_mean([2, 8])", "4"),
            # decimal
            ("decimal.Decimal('1.1') + decimal.Decimal('2.2')", "3.3"),
            ("decimal.getcontext().prec", "28"),
            # re (Pattern objects fine — their dunders are AST-blocked)
            ("re.compile(r'\\d+').findall('a1b22c333')", "333"),
            ("re.match(r'\\d+', '123abc').group()", "123"),
            ("re.search(r'b', 'abc').start()", "1"),
            ("re.sub(r'\\d', 'X', 'a1b2')", "aXbX"),
            ("re.split(r',', 'a,b,c')", "['a', 'b', 'c']"),
            # fractions (last-expression result shows repr; print() shows 1/2)
            ("print(fractions.Fraction(1, 3) + fractions.Fraction(1, 6))", "1/2"),
            # itertools
            ("list(itertools.chain([1], [2, 3]))", "[1, 2, 3]"),
            ("list(itertools.combinations([1, 2, 3], 2))", "(1, 2)"),
            # functools
            ("functools.reduce(lambda a, b: a + b, [1, 2, 3, 4])", "10"),
            ("functools.partial(lambda a, b: a + b, 10)(5)", "15"),
            # collections
            ("collections.OrderedDict([(1, 2)]).get(1)", "2"),
            ("collections.defaultdict(int)['x']", "0"),
            ("collections.Counter('aab').most_common(1)", "'a', 2"),
            ("collections.deque([1, 2, 3]).pop()", "3"),
            ("dict(collections.ChainMap({'a': 1}, {'b': 2}))", "'a': 1"),
            # copy
            ("copy.copy([1, 2, 3])", "[1, 2, 3]"),
            ("copy.deepcopy({'a': [1, 2]})", "'a': [1, 2]"),
            # calendar
            ("calendar.isleap(2024)", "True"),
            ("calendar.monthrange(2024, 2)[1]", "29"),
            # datetime (use a fixed date — no wall clock)
            ("datetime.date(2024, 1, 15).isoformat()", "2024-01-15"),
            (
                "(datetime.datetime(2024, 1, 2) - "
                "datetime.datetime(2024, 1, 1)).days",
                "1",
            ),
            # json
            ("json.loads('{\"a\": 1}')['a']", "1"),
            ("json.dumps({'a': 1})", '{"a": 1}'),
            # textwrap
            ("textwrap.dedent('    hi').strip()", "hi"),
            ("textwrap.shorten('a b c d', width=5)", "[...]"),
        ],
    )
    async def test_representative_call_per_module(
        self, code: str, expected: str
    ) -> None:
        tool = _make_tool()
        result = await _run_result(tool, code)
        assert result.success is True, f"{code}: {result.content}"
        assert expected in result.content, f"{code} -> {result.content}"

    @pytest.mark.asyncio
    async def test_namedtuple_still_creates_types(self) -> None:
        """``collections.namedtuple`` must still create working types after R6."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "Point = collections.namedtuple('Point', ['x', 'y'])\n"
            "p = Point(1, 2)\n"
            "print(p.x, p.y, p._asdict())",
        )
        assert result.success is True, result.content
        assert "1 2" in result.content

    @pytest.mark.asyncio
    async def test_deepcopy_of_nested_structure(self) -> None:
        """``copy.deepcopy`` must still deep-copy (independent nested lists)."""
        tool = _make_tool()
        result = await _run_result(
            tool,
            "a = [[1], [2]]\n"
            "b = copy.deepcopy(a)\n"
            "b[0].append(99)\n"
            "print(a, b)",
        )
        assert result.success is True, result.content
        assert "[[1], [2]] [[1, 99], [2]]" in result.content

    @pytest.mark.asyncio
    async def test_import_forms_still_work_for_every_module(self) -> None:
        """Bare ``import X`` and curated ``from X import <symbol>`` both work for
        every faceted module (the legit import surface is unchanged)."""
        tool = _make_tool()
        for mod in _ALL_FACETED_MODULES:
            result = await _run_result(tool, f"import {mod}")
            assert result.success, f"import {mod}: {result.content}"
        # A representative curated from-import per a couple of modules.
        for code in (
            "from statistics import mean",
            "from collections import OrderedDict, Counter, defaultdict",
            "from functools import reduce, partial, lru_cache",
            "from fractions import Fraction",
            "from itertools import chain, combinations",
            "from datetime import datetime, date, timedelta, timezone",
            "from decimal import Decimal, getcontext",
            "from collections.abc import Mapping",
        ):
            result = await _run_result(tool, code)
            assert result.success, f"{code}: {result.content}"


# ---------------------------------------------------------------------------
# SECURITY — review R7: extra_modules pandas/numpy route through the SAFE facade
#            (CRITICAL) + module-name allowlist (MEDIUM)
# ---------------------------------------------------------------------------


class TestR7FacadedRootRoutingViaExtraModules:
    """CRITICAL (review R7): pandas/numpy requested via ``extra_modules`` (or
    ``allowed_modules``) WITHOUT ``enable_dataframes=True`` MUST be routed through
    the SAFE dataframe facade — never the generic ``build_stdlib_facade`` (which
    copied ``read_pickle``/``np.load``/``.ctypes`` — a confirmed unpickle /
    native-libc RCE). So ``PythonComputeTool(extra_modules=["numpy","pandas"])``
    is as safe as ``enable_dataframes=True``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # pandas unpickle-RCE + file/network reach (module-level — facade omits).
            'import pandas as pd\npd.read_pickle("/tmp/x.pkl")',
            'import pandas as pd\npd.read_csv("/etc/passwd")',
            "import pandas as pd\npd.eval('1+1')",
            # numpy unpickle-RCE / file reach (AST denylist on instance+module name).
            'import numpy as np\nnp.load("/tmp/x")',
            'import numpy as np\nnp.load("/tmp/x", allow_pickle=True)',
            'import numpy as np\nnp.fromfile("/tmp/x")',
            # pandas/numpy instance-method writers (AST denylist).
            'import pandas as pd\npd.DataFrame({"a": [1]}).to_pickle("/tmp/x")',
            'import numpy as np\nnp.array([1]).tofile("/tmp/x")',
            # ndarray native-libc bridge (AST denylist).
            "import numpy as np\nnp.array([1]).ctypes",
            "import numpy as np\nnp.array([1]).ctypes.data",
            # The from-import / dotted-submodule escapes (R2 / dotted fix).
            "from numpy.ctypeslib import load_library",
            "import numpy.ctypeslib as c\nc.ctypes.CDLL(None).system(b'x')",
            "from numpy import ctypeslib",
            "from pandas import io",
            "import pandas.io as pio\npio.pickle.read_pickle('/tmp/x.pkl')",
        ],
    )
    async def test_pickle_and_io_reach_blocked_without_enable_dataframes(
        self, code: str
    ) -> None:
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        tool = _make_tool(extra_modules=["numpy", "pandas"])
        result = await _run_result(tool, code)
        assert result.success is False, code

    @pytest.mark.asyncio
    async def test_positive_dataframe_ops_still_work(self) -> None:
        """Routing pandas/numpy through the safe facade must NOT break the
        legitimate in-memory compute surface."""
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        tool = _make_tool(extra_modules=["numpy", "pandas"])
        out_np = await _run_result(
            tool, "import numpy as np\nprint(np.array([1, 2, 3]).mean())"
        )
        assert out_np.success
        assert "2.0" in out_np.content
        out_pd = await _run_result(
            tool,
            'import pandas as pd\nprint(pd.DataFrame({"a": [1, 2, 3]})["a"].sum())',
        )
        assert out_pd.success
        assert "6" in out_pd.content

    @pytest.mark.asyncio
    async def test_extra_modules_numpy_only_does_not_add_pandas(self) -> None:
        """``extra_modules=["numpy"]`` exposes numpy (safe facade) but NOT pandas
        — the dataframe-safe routing injects only the requested facaded root."""
        pytest.importorskip("numpy")
        tool = _make_tool(extra_modules=["numpy"])
        assert "numpy" in tool._allowed_modules  # type: ignore[attr-defined]
        assert "pandas" not in tool._allowed_modules  # type: ignore[attr-defined]
        # numpy works through the safe facade; pickle reach still blocked.
        ok = await _run_result(tool, "import numpy as np\nprint(np.array([1, 2]).sum())")
        assert ok.success
        blocked = await _run_result(
            tool, 'import numpy as np\nnp.load("/tmp/x", allow_pickle=True)'
        )
        assert blocked.success is False

    @pytest.mark.asyncio
    async def test_allowed_modules_facaded_root_also_safe(self) -> None:
        """A facaded root via ``allowed_modules`` (not just ``extra_modules``)
        also receives the SAFE treatment."""
        pytest.importorskip("numpy")
        tool = _make_tool(allowed_modules=["math", "numpy"])
        blocked = await _run_result(
            tool, 'import numpy as np\nnp.load("/tmp/x", allow_pickle=True)'
        )
        assert blocked.success is False
        ok = await _run_result(tool, "import numpy as np\nprint(np.array([1, 2]).sum())")
        assert ok.success


class TestR7ModuleNameAllowlist:
    """MEDIUM (review R7): ``extra_modules``/``allowed_modules`` accept ONLY the
    vetted allowlist (pandas/numpy + the stdlib whitelist). An arbitrary module
    name (``os``/``subprocess``/``socket``/``requests``/``uuid``) is rejected at
    construction — closing ``extra_modules=["os"]`` -> in-sandbox ``os.system``."""

    @pytest.mark.parametrize(
        "bad", ["os", "subprocess", "socket", "requests", "sys", "shutil", "uuid"]
    )
    def test_extra_modules_arbitrary_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            _make_tool(extra_modules=[bad])

    @pytest.mark.parametrize("bad", ["os", "subprocess", "socket", "requests"])
    def test_allowed_modules_arbitrary_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            _make_tool(allowed_modules=[bad])

    def test_rejection_message_lists_allowed(self) -> None:
        """The error names the offending module and the allowed set."""
        with pytest.raises(ValueError) as exc:
            _make_tool(extra_modules=["os"])
        msg = str(exc.value)
        assert "'os'" in msg
        assert "math" in msg  # a representative allowlisted stdlib module
        assert "numpy" in msg  # a facaded root

    def test_multiple_bad_modules_all_reported(self) -> None:
        with pytest.raises(ValueError) as exc:
            _make_tool(extra_modules=["os", "subprocess"])
        msg = str(exc.value)
        assert "'os'" in msg
        assert "'subprocess'" in msg

    def test_allowlisted_stdlib_extra_module_accepted(self) -> None:
        """A stdlib-whitelist member is accepted via extra_modules (it is in
        ``_ALLOWED_IMPORT_MODULES`` already, but passing it explicitly is fine)."""
        tool = _make_tool(extra_modules=["json"])
        assert "json" in tool._allowed_modules  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_os_system_unreachable_via_extra_modules(self) -> None:
        """End-to-end: the MEDIUM RCE (``extra_modules=["os"]`` -> os.system) is
        impossible — the tool cannot even be constructed."""
        with pytest.raises(ValueError, match="not permitted"):
            _make_tool(extra_modules=["os"])


class TestR7ScaffoldConfigIsSafe:
    """The Designer scaffold (``agent_designer/assets.py``) must produce a SAFE
    compute tool: it now requests ``enable_dataframes=True`` (not raw
    ``extra_modules:["pandas","numpy"]``), and EITHER authored config must build a
    sandbox with no read_pickle/np.load reach."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            # The new (honest) scaffold config.
            {"enable_dataframes": True},
            # The legacy/raw form — still safe by construction (Fix 1).
            {"extra_modules": ["pandas", "numpy"]},
        ],
    )
    async def test_scaffold_compute_tool_has_no_pickle_reach(
        self, config: dict[str, object]
    ) -> None:
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="compute", kind="compute", config=config)
        tool = await factory.create(decl, ToolFactoryContext())
        assert isinstance(tool, PythonComputeTool)
        args = tool.validate_arguments(
            {"code": 'import pandas as pd\npd.read_pickle("/tmp/x.pkl")'}
        )
        result = await tool.execute(args, _ctx())
        assert result.success is False
        # And positive ops work.
        ok_args = tool.validate_arguments(
            {"code": "import numpy as np\nprint(np.array([1, 2, 3]).sum())"}
        )
        ok = await tool.execute(ok_args, _ctx())
        assert ok.success
        assert "6" in ok.content


class TestR7DefaultPathByteIdentical:
    """The default (no extra_modules, no enable_dataframes) sandbox is unchanged
    — the allowlist + routing only activate when modules are explicitly listed."""

    def test_default_allowed_modules_unchanged(self) -> None:
        from databricks_deep_research.tools.builtins.compute import (
            _ALLOWED_IMPORT_MODULES,
        )

        tool = _make_tool()
        # Exactly the stdlib whitelist (no pandas/numpy, no dataframe routing).
        assert set(tool._allowed_modules) == set(  # type: ignore[attr-defined]
            _ALLOWED_IMPORT_MODULES
        )

    def test_default_no_dataframe_method_block(self) -> None:
        """The dataframe AST denylist is inert by default (no facaded root)."""
        tool = _make_tool()
        assert tool._blocked_attrs == frozenset()  # type: ignore[attr-defined]
        assert tool._block_facaded_submodules is False  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_default_pandas_numpy_unavailable(self) -> None:
        """Without an explicit request, pandas/numpy are NOT importable."""
        tool = _make_tool()
        for code in ("import pandas", "import numpy"):
            result = await _run_result(tool, code)
            assert result.success is False
            assert "not available" in result.content.lower()

"""Security + functionality tests for the safe pandas/numpy compute subset.

Covers spec §5.2: ``import pandas``/``import numpy`` resolve to curated facades,
the DataFrame/array ergonomics work, every file/network/pickle/eval reach is
blocked (module-level omission OR AST denylist), the AST dunder guard still
blocks escapes WITH the facades present, and the default (no-dataframe) path is
byte-identical.

Round-4 structural invariant (HARD-gate): NO LIVE MODULE OBJECT IS REACHABLE
FROM THE SANDBOX.  numpy's ``random``/``linalg``/``fft`` are curated SUB-FACADES
(not the real submodules, which re-exported ``.os``/``.ctypeslib``/``.overrides``/
``.np`` — the proven R3 escapes); matplotlib is DROPPED entirely; ``.ctypes`` is
AST-denylisted; and a build-time assertion fails fast if any facade re-leaks a
live module.
"""

from __future__ import annotations

import types

import pytest

from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.builtins.compute_dataframe import (
    BLOCKED_DATAFRAME_METHODS,
    FACADED_ROOTS,
    NUMPY_FACADE_SYMBOLS,
    NUMPY_FFT_SUBFACADE_SYMBOLS,
    NUMPY_LINALG_SUBFACADE_SYMBOLS,
    NUMPY_RANDOM_SUBFACADE_SYMBOLS,
    PANDAS_FACADE_SYMBOLS,
    SANDBOX_ALLOWED_DOTTED_IMPORTS,
    _assert_no_live_module,
    assert_no_live_module,
    build_dataframe_modules,
    build_numpy_facade,
    build_pandas_facade,
    build_stdlib_facade,
)
from databricks_deep_research.tools.protocol import ToolContext

pytest.importorskip("pandas")
pytest.importorskip("numpy")


def _make_tool(**kwargs: object) -> PythonComputeTool:
    return PythonComputeTool(**kwargs)  # type: ignore[arg-type]


def _ctx() -> ToolContext:
    return ToolContext(query="test")


async def _run_result(tool: PythonComputeTool, code: str):  # type: ignore[no-untyped-def]
    args = tool.validate_arguments({"code": code})
    return await tool.execute(args, _ctx())


async def _run(tool: PythonComputeTool, code: str) -> str:
    result = await _run_result(tool, code)
    return result.content


# ---------------------------------------------------------------------------
# Facade construction
# ---------------------------------------------------------------------------


class TestFacadeConstruction:
    def test_pandas_facade_exposes_safe_symbols(self) -> None:
        facade = build_pandas_facade()
        assert facade is not None
        for name in ("DataFrame", "Series", "concat", "merge", "pivot_table"):
            assert hasattr(facade, name), name

    def test_pandas_facade_omits_readers(self) -> None:
        facade = build_pandas_facade()
        assert facade is not None
        for name in (
            "read_pickle",
            "read_csv",
            "read_sql",
            "read_parquet",
            "read_excel",
            "read_json",
            "read_html",
            "ExcelFile",
            "HDFStore",
            "eval",
        ):
            assert not hasattr(facade, name), f"facade must omit pandas.{name}"

    def test_numpy_facade_exposes_safe_symbols(self) -> None:
        facade = build_numpy_facade()
        assert facade is not None
        for name in ("array", "mean", "std", "concatenate", "where", "ndarray"):
            assert hasattr(facade, name), name

    def test_numpy_facade_omits_io(self) -> None:
        facade = build_numpy_facade()
        assert facade is not None
        for name in (
            "load",
            "loadtxt",
            "fromfile",
            "genfromtxt",
            "save",
            "savez",
            "savetxt",
            "memmap",
            "ctypeslib",
            "f2py",
            "lib",
        ):
            assert not hasattr(facade, name), f"facade must omit numpy.{name}"

    def test_build_dataframe_modules_includes_pandas_numpy(self) -> None:
        modules = build_dataframe_modules()
        assert "pandas" in modules
        assert "numpy" in modules

    def test_build_dataframe_modules_excludes_matplotlib(self) -> None:
        """Round 4: matplotlib is dropped from the sandbox (charts deferred)."""
        modules = build_dataframe_modules()
        assert "matplotlib" not in modules

    def test_matplotlib_not_a_facaded_root(self) -> None:
        """matplotlib is no longer a facaded root and has no dotted exception."""
        assert "matplotlib" not in FACADED_ROOTS
        assert frozenset({"pandas", "numpy"}) == FACADED_ROOTS
        assert len(SANDBOX_ALLOWED_DOTTED_IMPORTS) == 0

    def test_denylist_contains_known_reaches(self) -> None:
        for name in (
            "to_pickle",
            "to_csv",
            "to_sql",
            "eval",
            "query",
            "tofile",
            "dump",
            "ctypes",  # round 4: ndarray .ctypes native-libc bridge
        ):
            assert name in BLOCKED_DATAFRAME_METHODS

    def test_facade_symbol_lists_nonempty(self) -> None:
        assert len(PANDAS_FACADE_SYMBOLS) > 10
        assert len(NUMPY_FACADE_SYMBOLS) > 30

    # -- Sub-facade construction (round 4) ----------------------------------

    def test_numpy_submodules_are_subfacades_not_real_modules(self) -> None:
        """``np.random``/``np.linalg``/``np.fft`` are curated sub-facades whose
        ``__name__`` is ``numpy.<sub>`` but which carry NONE of the real
        submodule's live-module attributes (``.os``/``.np``/``.ctypeslib``/
        ``.overrides``/internal modules)."""
        facade = build_numpy_facade()
        assert facade is not None
        for sub_name in ("random", "linalg", "fft"):
            sub = getattr(facade, sub_name)
            assert isinstance(sub, types.ModuleType)
            assert sub.__name__ == f"numpy.{sub_name}"
            # The R3 live-module reaches must be ABSENT.
            for leaked in ("os", "sys", "np", "ctypeslib", "ctypes", "overrides"):
                assert not hasattr(sub, leaked), f"numpy.{sub_name}.{leaked} leaked"

    def test_numpy_random_subfacade_exposes_safe_prng(self) -> None:
        facade = build_numpy_facade()
        assert facade is not None
        for name in ("rand", "randn", "randint", "normal", "default_rng", "choice"):
            assert hasattr(facade.random, name), name

    def test_numpy_linalg_subfacade_exposes_safe_ops(self) -> None:
        facade = build_numpy_facade()
        assert facade is not None
        for name in ("inv", "solve", "det", "eig", "svd", "norm", "matrix_rank"):
            assert hasattr(facade.linalg, name), name

    def test_numpy_fft_subfacade_exposes_safe_transforms(self) -> None:
        facade = build_numpy_facade()
        assert facade is not None
        for name in ("fft", "ifft", "rfft", "fft2", "fftfreq"):
            assert hasattr(facade.fft, name), name

    def test_subfacade_internal_real_submodules_unreachable(self) -> None:
        """The proven R3 internal-submodule reaches are gone from the sub-facades:
        ``np.random._bounded_integers`` and ``np.linalg.linalg`` are AbsentError."""
        facade = build_numpy_facade()
        assert facade is not None
        assert not hasattr(facade.random, "_bounded_integers")
        assert not hasattr(facade.linalg, "linalg")
        assert not hasattr(facade.fft, "_pocketfft")

    def test_subfacade_symbol_lists_nonempty(self) -> None:
        assert len(NUMPY_RANDOM_SUBFACADE_SYMBOLS) > 5
        assert len(NUMPY_LINALG_SUBFACADE_SYMBOLS) > 5
        assert len(NUMPY_FFT_SUBFACADE_SYMBOLS) > 5

    # -- Build-time no-ModuleType structural assertion (round 4) ------------

    def test_facades_contain_zero_module_typed_attributes(self) -> None:
        """The CORE structural invariant: NO attribute on any facade (top-level
        AND recursively one level into the sub-facades) is a live module — except
        the curated sub-facades themselves (which carry only safe callables)."""
        for facade in (build_numpy_facade(), build_pandas_facade()):
            assert facade is not None
            for attr_name in dir(facade):
                if attr_name.startswith("__"):
                    continue
                attr_value = getattr(facade, attr_name)
                if not isinstance(attr_value, types.ModuleType):
                    continue
                # A module-typed attr is allowed only if it is a curated
                # sub-facade — which must itself contain ZERO module-typed attrs.
                for sub_attr_name in dir(attr_value):
                    if sub_attr_name.startswith("__"):
                        continue
                    sub_value = getattr(attr_value, sub_attr_name)
                    assert not isinstance(sub_value, types.ModuleType), (
                        f"{facade.__name__}.{attr_name}.{sub_attr_name} "
                        "is a live module"
                    )

    def test_build_time_assertion_passes_on_real_facades(self) -> None:
        """``_assert_no_live_module`` must NOT raise on the genuine facades."""
        np_facade = build_numpy_facade()
        pd_facade = build_pandas_facade()
        assert np_facade is not None
        assert pd_facade is not None
        _assert_no_live_module(np_facade)
        _assert_no_live_module(pd_facade)

    def test_build_time_assertion_catches_top_level_module_leak(self) -> None:
        """A regression that re-exports the REAL numpy submodule (which carries
        internal modules) fails fast at construction."""
        import numpy as real_numpy

        facade = build_numpy_facade()
        assert facade is not None
        facade.random = real_numpy.random  # re-leak the REAL submodule
        with pytest.raises(RuntimeError, match="live module"):
            _assert_no_live_module(facade)

    def test_build_time_assertion_catches_deep_module_leak(self) -> None:
        """A regression that leaks ``os`` one level into a sub-facade fails fast."""
        import os

        facade = build_numpy_facade()
        assert facade is not None
        facade.random.os = os  # leak a live module onto the sub-facade
        with pytest.raises(RuntimeError, match="live module"):
            _assert_no_live_module(facade)


# ---------------------------------------------------------------------------
# Functionality — the subset must actually work
# ---------------------------------------------------------------------------


class TestPandasFunctionality:
    @pytest.mark.asyncio
    async def test_dataframe_construction_and_sum(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            'import pandas as pd\ndf = pd.DataFrame({"a": [1, 2, 3]})\nprint(df["a"].sum())',
        )
        assert "6" in out

    @pytest.mark.asyncio
    async def test_groupby_aggregation(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            'import pandas as pd\n'
            'df = pd.DataFrame({"k": ["x", "x", "y"], "v": [1, 2, 3]})\n'
            'print(df.groupby("k")["v"].sum().to_dict())',
        )
        assert "'x': 3" in out
        assert "'y': 3" in out

    @pytest.mark.asyncio
    async def test_merge(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            'import pandas as pd\n'
            'a = pd.DataFrame({"k": [1], "x": [2]})\n'
            'b = pd.DataFrame({"k": [1], "y": [3]})\n'
            'print(pd.merge(a, b, on="k").to_dict("records"))',
        )
        assert "'x': 2" in out
        assert "'y': 3" in out

    @pytest.mark.asyncio
    async def test_pivot_table(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            'import pandas as pd\n'
            'df = pd.DataFrame({"a": ["x", "x"], "b": [1, 2]})\n'
            'print(pd.pivot_table(df, index="a", values="b", aggfunc="sum").to_dict())',
        )
        assert "'x': 3" in out

    @pytest.mark.asyncio
    async def test_from_pandas_import_blocked_use_attribute(self) -> None:
        """R2 escape fix: ``from pandas import <name>`` is now rejected (the
        IMPORT_FROM ``sys.modules`` submodule fallback re-opened the real
        ``pandas.io``); use ``import pandas as pd`` + attribute access instead."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            'from pandas import DataFrame\nprint(DataFrame({"a": [1, 2]})["a"].sum())',
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()
        # The supported equivalent works.
        out = await _run(
            tool,
            'import pandas as pd\nprint(pd.DataFrame({"a": [1, 2]})["a"].sum())',
        )
        assert "3" in out

    @pytest.mark.asyncio
    async def test_describe(self) -> None:
        """describe() exercises numpy reduction internals through pandas — must
        still work after the dotted-submodule escape fix."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import pandas as pd\n"
            'df = pd.DataFrame({"a": [1, 2, 3]})\n'
            'print(df.describe().loc["mean", "a"])',
        )
        assert "2.0" in out

    @pytest.mark.asyncio
    async def test_groupby_mean(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import pandas as pd\n"
            'df = pd.DataFrame({"k": ["x", "x", "y"], "v": [1, 3, 5]})\n'
            'print(df.groupby("k")["v"].mean().to_dict())',
        )
        assert "'x': 2.0" in out
        assert "'y': 5.0" in out


class TestNumpyFunctionality:
    @pytest.mark.asyncio
    async def test_array_mean_method(self) -> None:
        """The ndarray METHOD path triggers numpy's internal lazy import."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(tool, "import numpy as np\nprint(np.array([1.0, 2, 3]).mean())")
        assert "2.0" in out

    @pytest.mark.asyncio
    async def test_array_std_method(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool, "import numpy as np\nprint(round(np.array([1.0, 2, 3, 4]).std(), 4))"
        )
        assert "1.118" in out

    @pytest.mark.asyncio
    async def test_mean_function(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(tool, "import numpy as np\nprint(np.mean([1.0, 2, 3]))")
        assert "2.0" in out

    @pytest.mark.asyncio
    async def test_linalg_norm(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(tool, "import numpy as np\nprint(np.linalg.norm([3, 4]))")
        assert "5" in out

    @pytest.mark.asyncio
    async def test_linalg_inv(self) -> None:
        """Sub-facade ``np.linalg.inv`` (round 4 curated sub-facade) works."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import numpy as np\n"
            "m = np.linalg.inv(np.eye(2))\nprint(int(m[0, 0]), int(m[1, 1]))",
        )
        assert "1 1" in out

    @pytest.mark.asyncio
    async def test_fft_fft(self) -> None:
        """Sub-facade ``np.fft.fft`` (round 4 curated sub-facade) works."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool, "import numpy as np\nprint(int(np.fft.fft([1, 2, 3, 4])[0].real))"
        )
        assert "10" in out

    @pytest.mark.asyncio
    async def test_random_default_rng_integers(self) -> None:
        """Sub-facade ``np.random.default_rng`` (round 4) works deterministically."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import numpy as np\n"
            "vals = np.random.default_rng(0).integers(0, 10, 5)\n"
            "print(len(vals), all(0 <= v < 10 for v in vals))",
        )
        assert "5 True" in out

    @pytest.mark.asyncio
    async def test_array_basic_stats(self) -> None:
        """The spec positive trio: mean/std/sum on a constructed array."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import numpy as np\n"
            "a = np.array([1, 2, 3])\n"
            "print(float(a.mean()), int(a.sum()))",
        )
        assert "2.0" in out
        assert "6" in out

    @pytest.mark.asyncio
    async def test_from_numpy_import_blocked_use_attribute(self) -> None:
        """R2 escape fix: ``from numpy import <name>`` is now rejected (the
        IMPORT_FROM ``sys.modules`` submodule fallback re-resolved the real
        ``numpy.ctypeslib``); use ``import numpy as np`` + attribute access."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, "from numpy import array\nprint(array([1, 2, 3]).sum())"
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()
        # The supported equivalent works.
        out = await _run(tool, "import numpy as np\nprint(np.array([1, 2, 3]).sum())")
        assert "6" in out

    @pytest.mark.asyncio
    async def test_reduction_methods_after_dotted_block(self) -> None:
        """Regression for the escape fix: removing the dotted-traversal branch
        must NOT break numpy's internal ``from numpy._core import _methods``
        (fired by mean/std/var). These are served from the pre-warmed
        ``sys.modules`` cache, never through the sandbox import statement."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import numpy as np\n"
            "a = np.array([1.0, 2, 3, 4])\n"
            "print(round(a.mean(), 4), round(a.std(), 4), "
            "round(a.var(), 4), int(a.sum()))",
        )
        assert "2.5" in out  # mean
        assert "1.118" in out  # std
        assert "1.25" in out  # var
        assert "10" in out  # sum

    @pytest.mark.asyncio
    async def test_var_method(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool, "import numpy as np\nprint(round(np.array([1.0, 2, 3]).var(), 4))"
        )
        assert "0.6667" in out


# ---------------------------------------------------------------------------
# SECURITY — module-level reaches blocked (facade omission)
# ---------------------------------------------------------------------------


class TestModuleLevelReachesBlocked:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expr",
        [
            'pd.read_pickle("/tmp/x")',
            'pd.read_csv("/etc/passwd")',
            "pd.read_sql('SELECT 1', None)",
            'pd.read_parquet("/tmp/x")',
            'pd.read_excel("/tmp/x")',
            'pd.read_json("/tmp/x")',
            'pd.read_html("/tmp/x")',
            "pd.eval('1+1')",
        ],
    )
    async def test_pandas_reader_blocked(self, expr: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, f"import pandas as pd\n{expr}")
        assert result.success is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expr",
        [
            'np.load("/tmp/x")',
            'np.load("/tmp/x", allow_pickle=True)',
            'np.fromfile("/tmp/x")',
            'np.genfromtxt("/tmp/x")',
            'np.loadtxt("/tmp/x")',
            'np.save("/tmp/x", [1])',
            "np.ctypeslib",
            "np.f2py",
        ],
    )
    async def test_numpy_io_blocked(self, expr: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, f"import numpy as np\n{expr}")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_from_pandas_import_read_csv_blocked(self) -> None:
        """The critical from-import escape: must NOT bind read_csv."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "from pandas import read_csv")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_from_numpy_import_load_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "from numpy import load")
        assert result.success is False


# ---------------------------------------------------------------------------
# SECURITY — dotted-submodule sandbox escape blocked (spec §5.2 escape fix)
# ---------------------------------------------------------------------------


class TestDottedSubmoduleEscapeBlocked:
    """The CRITICAL fix: a sandbox-authored DOTTED import rooted at a facaded
    library must be rejected at parse time, defeating every proven escape that
    walked the REAL backing submodule (unpickle-RCE / arbitrary file read+write /
    native-libc RCE / SSRF)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # pandas.io family — the proven RCE / file / SSRF / DB escapes.
            "import pandas.io",
            "import pandas.io as pio",
            "import pandas.io.pickle",
            "import pandas.io.common",
            "import pandas.io.parsers",
            "import pandas.io.html",
            "import pandas.io.sql",
            "from pandas.io.parsers import read_csv",
            "from pandas.io import pickle",
            "from pandas.io.pickle import read_pickle",
            "from pandas.io.common import get_handle",
            "from pandas.io.html import read_html",
            "from pandas.io.sql import read_sql",
            # numpy submodules — ctypeslib (native libc), lib/core (IO internals).
            "import numpy.ctypeslib",
            "import numpy.ctypeslib as c",
            "import numpy.lib",
            "import numpy.core",
            "import numpy._core",
            "from numpy.ctypeslib import ctypes",
            "from numpy.lib import npyio",
            "from numpy._core import _methods",
            "from numpy.core.multiarray import fromfile",
        ],
    )
    async def test_dotted_facade_import_blocked(self, code: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_pandas_io_pickle_read_pickle_unreachable(self) -> None:
        """End-to-end: the full unpickle-RCE chain is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "import pandas.io as pio\npio.pickle.read_pickle('/tmp/x.pkl')",
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_pandas_io_common_get_handle_unreachable(self) -> None:
        """Arbitrary file WRITE via get_handle is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "import pandas.io as pio\npio.common.get_handle('/tmp/x', 'w')",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_pandas_io_parsers_read_csv_unreachable(self) -> None:
        """Arbitrary file READ via read_csv is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "from pandas.io.parsers import read_csv\nread_csv('/etc/passwd')",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_pandas_io_html_read_html_unreachable(self) -> None:
        """SSRF/network via read_html is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "import pandas.io as pio\npio.html.read_html('http://169.254.169.254/')",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_pandas_io_sql_read_sql_unreachable(self) -> None:
        """DB access via read_sql is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "import pandas.io as pio\npio.sql.read_sql('SELECT 1', None)",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_numpy_ctypeslib_libc_unreachable(self) -> None:
        """Native libc RCE via numpy.ctypeslib -> ctypes.CDLL is blocked."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "import numpy.ctypeslib as c\nc.ctypes.CDLL(None).system(b'echo pwned')",
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_bare_root_imports_still_work(self) -> None:
        """The sole allowed import form (bare root → facade) is unaffected;
        symbols are reached as attributes (``np.array`` / ``pd.DataFrame``).
        ``from <root> import <name>`` is now rejected (R2 escape fix) — see
        :meth:`test_from_numpy_import_blocked_use_attribute`."""
        tool = _make_tool(enable_dataframes=True)
        for code in (
            "import numpy as np\nprint(np.array([1, 2, 3]).sum())",
            "import pandas as pd\nprint(pd.DataFrame({'a': [1, 2]})['a'].sum())",
            "import numpy\nprint(numpy.array([1, 2]).sum())",
            "import pandas\nprint(pandas.DataFrame({'a': [1]})['a'].sum())",
        ):
            result = await _run_result(tool, code)
            assert result.success is True, code

    @pytest.mark.asyncio
    async def test_dotted_facade_import_allowed_when_disabled(self) -> None:
        """The submodule block is inert with dataframes OFF — but the root is
        still unavailable (no facade), so it fails for the legacy reason."""
        tool = _make_tool()  # enable_dataframes defaults to False
        result = await _run_result(tool, "import pandas.io")
        assert result.success is False
        # Legacy message (module not available), NOT the new submodule block.
        assert "not available" in result.content.lower()


# ---------------------------------------------------------------------------
# SECURITY — R2 from-import sandbox escape blocked (spec §5.2, round 3)
# ---------------------------------------------------------------------------


class TestFromFacadedRootImportBlocked:
    """R2 CRITICAL #1: ``from <facaded-root> import <name>`` is rejected at parse
    time.  ``from numpy import ctypeslib`` did ``__import__('numpy',
    fromlist=['ctypeslib'])`` → the facade, then CPython's IMPORT_FROM
    ``getattr(facade, 'ctypeslib')`` raised AttributeError and FELL BACK to
    ``sys.modules['numpy.ctypeslib']`` — the REAL ``numpy.ctypeslib`` →
    ``ctypeslib.ctypes.CDLL(None).system(...)`` (RCE).  ``from pandas import io``
    re-opened R1's ``pandas.io`` pickle/file/SSRF/DB family."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # The proven RCE / facade-defeat forms.
            "from numpy import ctypeslib",
            "from numpy import ctypeslib as c",
            "from numpy import core",
            "from numpy import _core",
            "from numpy import lib",
            "from numpy import ctypeslib, core",
            "from pandas import io",
            "from pandas import io as pio",
            "from pandas import compat",
            # Even an otherwise-safe curated symbol via ``from`` is rejected
            # (the import form itself is the gate — no per-name allowlisting).
            "from numpy import array",
            "from pandas import DataFrame",
        ],
    )
    async def test_from_facaded_root_import_blocked(self, code: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_from_numpy_import_ctypeslib_rce_unreachable(self) -> None:
        """End-to-end: the full ``from numpy import ctypeslib`` RCE chain is
        blocked at the import (the IMPORT_FROM ``sys.modules`` fallback never
        fires because the statement is rejected at parse time)."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "from numpy import ctypeslib\n"
            "ctypeslib.ctypes.CDLL(None).system(b'echo pwned')",
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_from_pandas_import_io_pickle_unreachable(self) -> None:
        """End-to-end: ``from pandas import io`` re-opening the pickle/file/SSRF
        family is blocked at the import."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool,
            "from pandas import io\nio.pickle.read_pickle('/tmp/x.pkl')",
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()


class TestRealModuleAttrUnreachable:
    """R2 CRITICAL #2: the real backing module must NOT be reachable by name.
    The facade no longer carries it under ``__compute_real_module__`` (the build
    no longer stashes it), and that attribute name is in the AST dunder denylist
    as defense-in-depth, so ``np.__compute_real_module__.ctypeslib.ctypes.
    CDLL(None).system(...)`` is rejected at parse time."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            "import numpy as np\nnp.__compute_real_module__",
            "import pandas as pd\npd.__compute_real_module__",
            "import numpy as np\n"
            "np.__compute_real_module__.ctypeslib.ctypes.CDLL(None).system(b'x')",
            "import pandas as pd\npd.__compute_real_module__.io.pickle.read_pickle('/tmp/x')",
        ],
    )
    async def test_real_module_attr_blocked(self, code: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False
        assert "not allowed" in result.content.lower()

    def test_facade_does_not_carry_real_module(self) -> None:
        """The facade build no longer stashes the real backing module under any
        attribute (dead code after R1 + a proven hole)."""
        np_facade = build_numpy_facade()
        pd_facade = build_pandas_facade()
        assert np_facade is not None
        assert pd_facade is not None
        assert not hasattr(np_facade, "__compute_real_module__")
        assert not hasattr(pd_facade, "__compute_real_module__")

    def test_real_module_attr_in_dunder_denylist(self) -> None:
        """Defense-in-depth: the attribute name is in the AST dunder denylist so
        even a stray real-module handle is unreachable by name."""
        from databricks_deep_research.tools.builtins.compute import (
            _BLOCKED_DUNDER_ATTRS,
        )

        assert "__compute_real_module__" in _BLOCKED_DUNDER_ATTRS


# ---------------------------------------------------------------------------
# SECURITY — R3 live-module sandbox escape blocked (spec §5.2, round 4)
# ---------------------------------------------------------------------------


class TestLiveModuleEscapeBlocked:
    """HARD-gate round 4: the 8 proven R3 escapes — every one reached a LIVE real
    module through the numpy facade (re-exported real submodules) or matplotlib
    (the real module).  The structural fix (curated sub-facades + matplotlib drop
    + ``.ctypes`` denylist) makes ALL of them unavailable: a sub-facade carries no
    ``.os``/``.np``/``.ctypeslib``/``.overrides`` reach, matplotlib will not
    import, and ``.ctypes`` is rejected at parse time."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # R3 #1-4: reaching the real numpy submodules / their internals.
            "import numpy as np\nnp.linalg.linalg",
            "import numpy as np\nnp.linalg.overrides",
            "import numpy as np\nnp.random._bounded_integers",
            "import numpy as np\nnp.fft._pocketfft",
            # R3 #5-6: the full live-module RCE pivot chains.
            "import numpy as np\nnp.linalg.linalg.overrides.os.system('echo pwned')",
            "import numpy as np\n"
            "np.random._bounded_integers.np.ctypeslib.ctypes.CDLL(None).system(b'x')",
            # The .np / .os / .sys re-export reaches on the sub-facades.
            "import numpy as np\nnp.random.np.ctypeslib",
            "import numpy as np\nnp.linalg.os.system('x')",
            "import numpy as np\nnp.fft.np",
        ],
    )
    async def test_numpy_submodule_live_reach_blocked(self, code: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False, code

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # R3 #7: ndarray .ctypes native-libc bridge — AST-denylisted.
            "import numpy as np\nnp.array([1]).ctypes",
            "import numpy as np\nnp.array([1]).ctypes.data",
            "import numpy as np\nnp.zeros(3).ctypes",
            # Any object's .ctypes attribute access is rejected at parse time.
            "import numpy as np\nx = np.eye(2)\nx.ctypes",
        ],
    )
    async def test_ctypes_attribute_access_blocked(self, code: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False, code
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            # R3 #8: matplotlib is DROPPED — not importable at all (any form).
            "import matplotlib",
            "import matplotlib as mpl",
            "import matplotlib.pyplot",
            "import matplotlib.pyplot as plt",
            "from matplotlib import pyplot",
            "from matplotlib.pyplot import plot",
            # The plt.sys -> os pivot is moot because the import fails first.
            "import matplotlib.pyplot as plt\nplt.sys.modules['os'].system('echo pwned')",
        ],
    )
    async def test_matplotlib_not_importable(self, code: str) -> None:
        """matplotlib (and pyplot) cannot be imported into the sandbox at all —
        the ``plt.sys.modules['os']`` pivot surface is removed by construction."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False, code
        # Either "not available" (bare root) or "not allowed" (dotted) — both
        # mean the live matplotlib module never reaches the sandbox.
        msg = result.content.lower()
        assert "not available" in msg or "not allowed" in msg

    @pytest.mark.asyncio
    async def test_numpy_subfacade_positive_still_works(self) -> None:
        """Sanity: blocking the live reach did NOT break the curated functions."""
        tool = _make_tool(enable_dataframes=True)
        out = await _run(
            tool,
            "import numpy as np\n"
            "print(round(float(np.linalg.det(np.eye(2))), 4))\n"
            "print(int(np.fft.fft([1, 2, 3, 4])[0].real))\n"
            "print(len(np.random.default_rng(0).integers(0, 10, 5)))",
        )
        assert "1.0" in out  # det(I) == 1
        assert "10" in out  # sum of [1,2,3,4] real part of fft[0]
        assert "5" in out  # rng produced 5 integers


# ---------------------------------------------------------------------------
# SECURITY — instance-method reaches blocked (AST denylist)
# ---------------------------------------------------------------------------


class TestInstanceMethodReachesBlocked:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_call",
        [
            'to_pickle("/tmp/x")',
            'to_csv("/tmp/x")',
            "to_sql('t', None)",
            'to_parquet("/tmp/x")',
            'to_excel("/tmp/x")',
            'to_json("/tmp/x")',
            "eval('a + 1')",
            'query("a > 0")',
        ],
    )
    async def test_dataframe_writer_method_blocked(self, method_call: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, f'import pandas as pd\npd.DataFrame({{"a": [1]}}).{method_call}'
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_call",
        ['tofile("/tmp/x")', 'dump("/tmp/x")'],
    )
    async def test_ndarray_writer_method_blocked(self, method_call: str) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, f"import numpy as np\nnp.array([1]).{method_call}"
        )
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_internal_submodule_load_blocked_by_ast(self) -> None:
        """Even reaching numpy internals, a blocked method name is rejected."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, 'import numpy as np\nfrom numpy.lib import npyio\nnpyio.load("/tmp/x")'
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# SECURITY — AST escape guards intact WITH facades present
# ---------------------------------------------------------------------------


class TestEscapeGuardsIntactWithFacades:
    @pytest.mark.asyncio
    async def test_class_hierarchy_escape_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, "().__class__.__bases__[0].__subclasses__()"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_dataframe_dunder_class_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(
            tool, 'import pandas as pd\npd.DataFrame({"a": [1]}).__class__'
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_globals_escape_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "import numpy as np\nnp.array.__globals__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_open_still_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, 'open("/etc/passwd")')
        assert result.success is False

    @pytest.mark.asyncio
    async def test_eval_builtin_still_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "eval('1+1')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_os_import_still_blocked(self) -> None:
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "import os")
        assert result.success is False


# ---------------------------------------------------------------------------
# Matplotlib — DROPPED from the sandbox (round 4; in-sandbox charts DEFERRED)
# ---------------------------------------------------------------------------


class TestMatplotlibDropped:
    """Round 4 (spec §5.2): matplotlib is removed from the sandbox entirely.  Its
    real module exposed ``plt.sys`` (→ ``sys.modules['os']`` RCE) and a
    Figure/canvas pivot surface that cannot be safely faceted in this sandbox, so
    in-sandbox chart rendering is DEFERRED pending a safe matplotlib strategy.
    The data-analysis (pandas/numpy) subset still ships.  The old confined-savefig
    machinery (``_savefig_target_allowed`` / ``build_matplotlib_module``) is
    removed; these tests assert it is gone AND that matplotlib will not import."""

    def test_savefig_machinery_removed(self) -> None:
        """The confined-savefig helpers are removed (matplotlib is dropped)."""
        import databricks_deep_research.tools.builtins.compute_dataframe as cdf

        assert not hasattr(cdf, "_savefig_target_allowed")
        assert not hasattr(cdf, "build_matplotlib_module")
        assert not hasattr(cdf, "_install_confined_savefig")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            "import matplotlib",
            "import matplotlib.pyplot as plt",
            "from matplotlib import pyplot",
        ],
    )
    async def test_matplotlib_unavailable(self, code: str) -> None:
        """matplotlib cannot be imported into the sandbox in any form."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, code)
        assert result.success is False, code
        msg = result.content.lower()
        assert "not available" in msg or "not allowed" in msg


# ---------------------------------------------------------------------------
# Default path — byte-identical when dataframes disabled
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    @pytest.mark.asyncio
    async def test_pandas_unavailable_by_default(self) -> None:
        tool = _make_tool()  # enable_dataframes defaults to False
        result = await _run_result(tool, "import pandas")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_numpy_unavailable_by_default(self) -> None:
        tool = _make_tool()
        result = await _run_result(tool, "import numpy")
        assert result.success is False
        assert "not available" in result.content.lower()

    @pytest.mark.asyncio
    async def test_method_name_allowed_when_disabled(self) -> None:
        """The method denylist is inert when dataframes are off — a variable or
        attribute named like a blocked method is NOT rejected at parse time."""
        tool = _make_tool()
        # ``eval``/``query`` are in the denylist; as a bare name in dataframe
        # mode they would be rejected. With dataframes OFF the guard is inert,
        # so assigning/using such a name parses and runs fine.
        out = await _run(tool, "eval = 5\nquery = 10\nprint(eval + query)")
        assert "15" in out

    @pytest.mark.asyncio
    async def test_method_name_blocked_when_enabled(self) -> None:
        """Symmetric check: the same bare name IS rejected with dataframes on."""
        tool = _make_tool(enable_dataframes=True)
        result = await _run_result(tool, "eval = 5\nprint(eval)")
        assert result.success is False
        assert "not allowed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_default_arithmetic_unchanged(self) -> None:
        tool = _make_tool()
        out = await _run(tool, "import math\nprint(math.sqrt(16))")
        assert "4.0" in out


# ---------------------------------------------------------------------------
# Generic stdlib facade (security review R6)
# ---------------------------------------------------------------------------


class TestStdlibFacade:
    """``build_stdlib_facade`` curates an arbitrary module by introspection:
    copy every public non-module attribute, drop every module handle."""

    def test_drops_module_typed_attributes(self) -> None:
        """A module that re-exports other modules (``calendar.sys``,
        ``calendar.datetime``) loses those handles in the facade."""
        import calendar as real_calendar

        facade = build_stdlib_facade("calendar", real_calendar)
        # The re-exported live modules are gone.
        assert not hasattr(facade, "sys")
        assert not hasattr(facade, "datetime")
        # The legitimate public API is preserved.
        assert facade.isleap is real_calendar.isleap
        assert facade.monthrange is real_calendar.monthrange
        assert hasattr(facade, "Calendar")

    def test_drops_underscore_names(self) -> None:
        """Underscore-prefixed names (incl. single-underscore module handles like
        ``collections._sys``/``collections._collections_abc``) are excluded."""
        import collections as real_collections

        facade = build_stdlib_facade("collections", real_collections)
        assert not hasattr(facade, "_sys")
        assert not any(name.startswith("_") for name in facade.__all__)  # type: ignore[attr-defined]
        # Public containers preserved.
        for name in ("OrderedDict", "defaultdict", "Counter", "deque", "namedtuple"):
            assert hasattr(facade, name), name

    def test_extra_denied_omits_named_symbols(self) -> None:
        """A name in *extra_denied* is omitted even if public + non-module."""
        import math as real_math

        facade = build_stdlib_facade(
            "math", real_math, extra_denied=frozenset({"sqrt"})
        )
        assert not hasattr(facade, "sqrt")
        assert hasattr(facade, "log")  # other symbols unaffected

    def test_facade_is_modtype_named_correctly(self) -> None:
        import re as real_re

        facade = build_stdlib_facade("re", real_re)
        assert isinstance(facade, types.ModuleType)
        assert facade.__name__ == "re"
        # ``re.functools``/``re.enum``/``re.copyreg`` module handles removed.
        for leaked in ("functools", "enum", "copyreg"):
            assert not hasattr(facade, leaked), leaked

    def test_assert_no_live_module_passes_on_stdlib_facade(self) -> None:
        import fractions as real_fractions

        facade = build_stdlib_facade("fractions", real_fractions)
        # Must not raise — no module-typed attribute survives.
        assert_no_live_module(facade)
        _assert_no_live_module(facade)

    def test_assert_no_live_module_catches_injected_leak(self) -> None:
        """If a facade is tampered to carry a live module, the assertion fails."""
        import os
        import textwrap as real_textwrap

        facade = build_stdlib_facade("textwrap", real_textwrap)
        facade.leaked = os  # re-introduce a live module handle
        with pytest.raises(RuntimeError, match="live module"):
            assert_no_live_module(facade)

    @pytest.mark.parametrize(
        "module_name",
        [
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
        ],
    )
    def test_every_default_module_facade_is_module_free(
        self, module_name: str
    ) -> None:
        """STRUCTURAL invariant per default module: ZERO module-typed public
        attribute survives faceting."""
        import importlib

        real_module = importlib.import_module(module_name)
        facade = build_stdlib_facade(module_name, real_module)
        leaked = [
            name
            for name in dir(facade)
            if not name.startswith("_")
            and isinstance(getattr(facade, name, None), types.ModuleType)
        ]
        assert leaked == [], f"{module_name} leaks {leaked}"

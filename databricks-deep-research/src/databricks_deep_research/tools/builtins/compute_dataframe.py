"""Safe pandas / numpy facades for the compute sandbox (US-109 §5.2).

US-109 deferred enabling pandas/numpy in :class:`PythonComputeTool` because the
naive "add the real module to ``extra_modules``" approach exposes arbitrary
file / network / pickle reach through PLAIN attribute calls that the existing
AST dunder guard does NOT block:

    pandas.read_pickle(path)            # unpickle = arbitrary code execution
    pandas.read_csv(path) / read_sql()  # file / network read
    numpy.load(path, allow_pickle=True) # unpickle = arbitrary code execution
    numpy.fromfile(path) / genfromtxt() # file read
    df.to_pickle(path) / arr.tofile(p)  # file write / pickle write
    pd.eval(expr) / df.eval / df.query  # arbitrary expression execution

This module re-implements pandas/numpy as **curated facade modules** that
re-export ONLY a safe allowlist of symbols and OMIT every ``read_*`` / save /
load / pickle / SQL / eval entry point.  The facades are wired into the sandbox
import allowlist so ``import pandas`` / ``import numpy`` resolve to the FACADE,
never the raw module (see :func:`build_dataframe_modules`).

THE STRUCTURAL INVARIANT (spec §5.2, HARD-gate round 4)
-------------------------------------------------------
NO LIVE MODULE OBJECT IS REACHABLE FROM THE SANDBOX.  Three prior security
rounds each found a NEW escape *class* by whack-a-mole over forms; this module
stops patching forms and enforces one invariant *by construction*: the sandbox
can reach ONLY curated facades + safe callables, NEVER a live ``types.ModuleType``.

Round 3 proved that re-exporting a REAL backing module — even a "pure-compute"
one — re-opens the escape, because a real module transitively re-exports its own
imports (``os``/``sys``/``ctypes``):

    np.linalg.linalg.overrides.os.system(...)                  # real linalg submodule
    np.random._bounded_integers.np.ctypeslib.ctypes.CDLL(None) # real random submodule
    plt.sys.modules['os'].system(...)                          # real matplotlib module
    np.array([1]).ctypes                                        # ndarray .ctypes

The fix removes every live-module reach:

  1. numpy's ``random`` / ``linalg`` / ``fft`` submodules (previously re-exported
     as the REAL submodules — :data:`_RANDOM_SUBFACADE_SYMBOLS` etc.) are now
     CURATED SUB-FACADES exposing ONLY safe pure-compute callables.  A sub-facade
     is a fresh ``ModuleType`` with no ``.os`` / ``.sys`` / ``.ctypeslib`` /
     ``.overrides`` / ``.np`` attribute — same pattern as the top-level facade.
  2. matplotlib is DROPPED from the sandbox entirely (it exposed ``plt.sys`` and a
     Figure/canvas pivot surface that cannot be safely faceted here).  The chart
     seed skill's in-sandbox rendering is therefore DEFERRED pending a safe
     matplotlib strategy; the ``data-analysis`` (pandas/numpy) subset still ships.
  3. ``ctypes`` is added to the AST attribute denylist so ``arr.ctypes`` / any
     ``.ctypes`` access is rejected at parse time.
  4. :func:`build_dataframe_modules` asserts at construction that NO attribute on
     ANY facade (top-level AND sub-facades, recursively one level) is a
     ``types.ModuleType`` — so a regression that re-leaks a live module fails
     fast at build, structurally, instead of becoming the next escape.

Defense-in-depth — the facade allowlist is necessary but NOT sufficient on its
own, because instance methods (``df.to_pickle(...)`` / ``arr.tofile(...)``) are
reachable on objects the facade legitimately hands out (a ``DataFrame`` the user
constructs still carries ``to_pickle``).  Those are blocked by an AST
attribute-name denylist (:data:`BLOCKED_DATAFRAME_METHODS`, enforced in
``compute._validate_ast``).  Together: the module facade closes the module-level
reaches, the AST denylist closes the instance-method reaches, and the existing
dunder/import/getattr guards close the dynamic-attribute reaches.

Submodule-import gate (spec §5.2, sandbox-escape fix) — a curated top-level
facade is NOT sufficient against a sandbox-authored import that reaches a real
backing module.  TWO forms defeat the facade and were proven RCE:
``import pandas.io as pio`` / ``from pandas.io.parsers import read_csv`` /
``import numpy.ctypeslib`` (DOTTED imports), AND any ``from <facaded-root>
import <name>`` (R2 escape — ``from numpy import ctypeslib`` re-resolves the real
``numpy.ctypeslib`` via CPython's IMPORT_FROM ``sys.modules`` fallback because
the facade's ``__name__`` is ``"numpy"``; ``from pandas import io`` re-opens the
``pandas.io`` family).  The compute AST validator therefore REJECTS every
sandbox import rooted at a facaded lib (:data:`FACADED_ROOTS`) EXCEPT a bare-root
import (``import numpy`` → facade; symbols reached as attributes, e.g.
``np.array(...)``).  There are now NO dotted-import exceptions
(:data:`SANDBOX_ALLOWED_DOTTED_IMPORTS` is empty — matplotlib was the only
entry and is dropped).  A library's OWN internal lazy imports (e.g. numpy's
``from numpy._core import _methods`` fired by ``ndarray.mean()``) live in the
library's compiled bytecode — they are NEVER part of the sandbox's parsed AST,
so the AST block does not touch them; the sandbox ``__import__`` serves those
RUNTIME calls from ``sys.modules`` (pre-warmed here) without ever handing the
internal module object to user code.

Security posture: when in doubt, expose LESS.

Optional dependency: pandas is NOT a hard framework dependency.  numpy ships in
the framework ``search`` extra.  Both facades are built only when the real module
imports successfully; a missing library degrades to "module unavailable" in the
sandbox (a clean ``ImportError``), never a crash at tool construction.
"""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime stays dependency-free
    import numpy as _np_types  # noqa: F401
    import pandas as _pd_types  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCKED_DATAFRAME_METHODS",
    "FACADED_ROOTS",
    "NUMPY_FACADE_SYMBOLS",
    "NUMPY_FFT_SUBFACADE_SYMBOLS",
    "NUMPY_LINALG_SUBFACADE_SYMBOLS",
    "NUMPY_RANDOM_SUBFACADE_SYMBOLS",
    "PANDAS_FACADE_SYMBOLS",
    "SANDBOX_ALLOWED_DOTTED_IMPORTS",
    "assert_no_live_module",
    "build_dataframe_modules",
    "build_numpy_facade",
    "build_pandas_facade",
    "build_stdlib_facade",
]


# ---------------------------------------------------------------------------
# Submodule-import security policy (spec §5.2)
# ---------------------------------------------------------------------------

# Roots whose top-level surface is replaced by a CURATED FACADE (pandas/numpy).
# For these roots a sandbox-authored DOTTED import (``import pandas.io`` / ``from
# pandas.io.parsers import read_csv`` / ``import numpy.ctypeslib``) is REJECTED —
# it would reach a real backing submodule and defeat the facade (proven RCE /
# arbitrary file read/write / native-libc escape).  ANY ``from <root> import
# <name>`` is ALSO rejected (R2 escape: CPython's IMPORT_FROM falls back to
# ``sys.modules['<root>.<name>']`` — the REAL submodule — because the facade's
# ``__name__`` is ``<root>``).  ONLY a bare-root import (``import numpy`` →
# facade; symbols reached as attributes such as ``np.array(...)``) is allowed.
#
# matplotlib is NO LONGER a facaded root: it is dropped from the sandbox (round 4
# — its ``plt.sys`` / Figure-canvas pivot surface cannot be safely faceted here;
# in-sandbox chart rendering is deferred), so ``import matplotlib`` simply fails
# the sandbox's "module not available" check like any other unlisted module.
FACADED_ROOTS: frozenset[str] = frozenset({"pandas", "numpy"})

# The dotted ``import X`` statements the sandbox permits for a facaded root.
# EMPTY by design (spec §5.2, round 4): ``matplotlib.pyplot`` was the only entry
# and matplotlib is dropped.  Retained as a (now-empty) frozenset so the compute
# AST validator's exception check needs no shape change.  A NON-empty value here
# would re-introduce a dotted-import allowance — do not add one without a facade.
SANDBOX_ALLOWED_DOTTED_IMPORTS: frozenset[str] = frozenset()

# numpy reduction methods (``ndarray.mean()``/``.std()``/``.var()``) trigger a
# RUNTIME ``__import__('numpy._core._methods', fromlist=())`` from numpy's OWN
# bytecode every call. A bare ``import numpy`` already caches these, but we warm
# them explicitly at facade-build time so the sandbox import guard's
# "already-cached internal submodule" branch is GUARANTEED to hit (robust across
# numpy versions with lazier package init). These are NEVER exposed to user code.
_NUMPY_PREWARM_SUBMODULES: tuple[str, ...] = (
    "numpy._core._methods",
    "numpy._core.multiarray",
    "numpy._core.umath",
    "numpy._core._dtype",
)


# ---------------------------------------------------------------------------
# pandas facade allowlist
# ---------------------------------------------------------------------------

# Safe, module-level pandas symbols re-exported into the sandbox facade.
# Deliberately OMITS every ``read_*`` (read_pickle/read_csv/read_sql/read_parquet/
# read_excel/read_json/read_html/read_table/read_feather/read_orc/read_hdf/
# read_fwf/read_clipboard/read_gbq/read_sas/read_spss/read_stata/read_xml),
# ``ExcelFile``, ``HDFStore``, and ``eval`` (arbitrary expression execution).
PANDAS_FACADE_SYMBOLS: tuple[str, ...] = (
    # Core containers (DataFrame/Series carry to_pickle/to_csv/eval/query as
    # instance methods — those are blocked by the AST denylist, not here).
    "DataFrame",
    "Series",
    "Index",
    "MultiIndex",
    "RangeIndex",
    "DatetimeIndex",
    "CategoricalIndex",
    "IntervalIndex",
    "Categorical",
    "CategoricalDtype",
    # Scalars / missing sentinels.
    "Timestamp",
    "Timedelta",
    "Period",
    "Interval",
    "NA",
    "NaT",
    "IndexSlice",
    # Reshaping / combining (pure in-memory transforms).
    "concat",
    "merge",
    "merge_ordered",
    "merge_asof",
    "pivot",
    "pivot_table",
    "melt",
    "crosstab",
    "get_dummies",
    "from_dummies",
    "wide_to_long",
    "cut",
    "qcut",
    "factorize",
    "unique",
    "Grouper",
    "NamedAgg",
    # Type coercion / construction (in-memory only; no file/network).
    "to_datetime",
    "to_numeric",
    "to_timedelta",
    "date_range",
    "timedelta_range",
    "period_range",
    "interval_range",
    "bdate_range",
    "array",
    "isna",
    "notna",
    "isnull",
    "notnull",
)


# ---------------------------------------------------------------------------
# numpy facade allowlist
# ---------------------------------------------------------------------------

# Safe, module-level numpy symbols.  Deliberately OMITS every file/pickle entry
# point (load/loadtxt/fromfile/genfromtxt/fromregex/frombuffer/fromstring/
# save/savez/savez_compressed/savetxt/memmap/DataSource) and the submodule
# objects (lib/core/ctypeslib/f2py/testing) so arbitrary-format / ctypes / IO
# reach is unavailable.  ``random``/``linalg``/``fft`` are NOT in this list —
# they are added separately as CURATED SUB-FACADES (see :func:`build_numpy_facade`
# and :data:`_NUMPY_SUBFACADES`), NEVER as the real submodules (round 4: the real
# submodules re-export ``.os``/``.ctypeslib``/``.overrides``/``.np`` — a live
# module reach).
NUMPY_FACADE_SYMBOLS: tuple[str, ...] = (
    # Construction (in-memory).
    "array",
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "empty",
    "empty_like",
    "eye",
    "identity",
    "diag",
    "diagflat",
    "tri",
    "tril",
    "triu",
    "meshgrid",
    "indices",
    "repeat",
    "tile",
    # Shape / combine.
    "reshape",
    "ravel",
    "flatten",
    "concatenate",
    "stack",
    "hstack",
    "vstack",
    "dstack",
    "column_stack",
    "split",
    "array_split",
    "hsplit",
    "vsplit",
    "transpose",
    "swapaxes",
    "moveaxis",
    "expand_dims",
    "squeeze",
    "flip",
    "fliplr",
    "flipud",
    "roll",
    "rot90",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_to",
    "broadcast_arrays",
    # Selection / search.
    "where",
    "select",
    "choose",
    "take",
    "compress",
    "extract",
    "nonzero",
    "argwhere",
    "flatnonzero",
    "searchsorted",
    "clip",
    "unique",
    "in1d",
    "isin",
    "intersect1d",
    "union1d",
    "setdiff1d",
    "setxor1d",
    # Sorting / ordering.
    "sort",
    "argsort",
    "lexsort",
    "partition",
    "argpartition",
    "argmax",
    "argmin",
    "nanargmax",
    "nanargmin",
    # Reductions / statistics.
    "sum",
    "nansum",
    "prod",
    "nanprod",
    "cumsum",
    "nancumsum",
    "cumprod",
    "nancumprod",
    "mean",
    "nanmean",
    "average",
    "median",
    "nanmedian",
    "std",
    "nanstd",
    "var",
    "nanvar",
    "min",
    "nanmin",
    "amin",
    "max",
    "nanmax",
    "amax",
    "ptp",
    "percentile",
    "nanpercentile",
    "quantile",
    "nanquantile",
    "histogram",
    "histogram2d",
    "histogramdd",
    "bincount",
    "digitize",
    "corrcoef",
    "cov",
    "count_nonzero",
    # Element-wise math (ufuncs — pure compute).
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "mod",
    "remainder",
    "power",
    "abs",
    "absolute",
    "fabs",
    "sign",
    "negative",
    "reciprocal",
    "sqrt",
    "cbrt",
    "square",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "degrees",
    "radians",
    "floor",
    "ceil",
    "round",
    "around",
    "rint",
    "trunc",
    "fix",
    "modf",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "gcd",
    "lcm",
    "clip",
    "interp",
    "diff",
    "ediff1d",
    "gradient",
    "cross",
    "dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "tensordot",
    "einsum",
    "trace",
    "kron",
    # Logical / comparison.
    "all",
    "any",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "isnan",
    "isinf",
    "isfinite",
    "isclose",
    "allclose",
    "array_equal",
    "array_equiv",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    # Types / dtypes (for isinstance + astype targets).
    "ndarray",
    "dtype",
    "number",
    "integer",
    "floating",
    "signedinteger",
    "unsignedinteger",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "str_",
    "datetime64",
    "timedelta64",
    "iinfo",
    "finfo",
    "result_type",
    "promote_types",
    "can_cast",
    "nan",
    "inf",
    "pi",
    "e",
    "euler_gamma",
    "newaxis",
    "errstate",
    "seterr",
    "geterr",
)


# ---------------------------------------------------------------------------
# numpy SUB-FACADE allowlists (random / linalg / fft)
# ---------------------------------------------------------------------------

# Round 4 (spec §5.2): ``np.random`` / ``np.linalg`` / ``np.fft`` are exposed as
# CURATED SUB-FACADES — fresh ``ModuleType`` objects re-exporting ONLY the safe
# pure-compute callables below.  The REAL submodules are NEVER reached by sandbox
# code: they re-export their own imports (``random.np.ctypeslib``,
# ``linalg.linalg.overrides.os``) which were the proven R3 live-module escapes.
# A sub-facade carries no ``.os`` / ``.sys`` / ``.ctypeslib`` / ``.overrides`` /
# ``.np`` attribute, and the build-time no-ModuleType assertion enforces that.

# PRNG functions only — no IO, no module attributes.
NUMPY_RANDOM_SUBFACADE_SYMBOLS: tuple[str, ...] = (
    "rand",
    "randn",
    "randint",
    "random",
    "random_sample",
    "sample",
    "ranf",
    "normal",
    "uniform",
    "standard_normal",
    "standard_uniform",
    "beta",
    "binomial",
    "chisquare",
    "exponential",
    "gamma",
    "geometric",
    "gumbel",
    "laplace",
    "logistic",
    "lognormal",
    "poisson",
    "power",
    "rayleigh",
    "triangular",
    "wald",
    "weibull",
    "zipf",
    "choice",
    "shuffle",
    "permutation",
    "seed",
    "default_rng",
    "Generator",
    "RandomState",
    "PCG64",
    "MT19937",
)

# Linear-algebra functions only — pure compute, no IO/module attributes.
NUMPY_LINALG_SUBFACADE_SYMBOLS: tuple[str, ...] = (
    "inv",
    "pinv",
    "solve",
    "tensorsolve",
    "tensorinv",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "svd",
    "qr",
    "cholesky",
    "norm",
    "cond",
    "matrix_rank",
    "lstsq",
    "slogdet",
    "matrix_power",
    "multi_dot",
    "LinAlgError",
)

# FFT functions only — pure compute, no IO/module attributes.
NUMPY_FFT_SUBFACADE_SYMBOLS: tuple[str, ...] = (
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
)

# Map of {numpy submodule name: curated symbol allowlist}.  Each becomes a fresh
# ``ModuleType`` sub-facade replacing the REAL submodule on the numpy facade.
_NUMPY_SUBFACADES: dict[str, tuple[str, ...]] = {
    "random": NUMPY_RANDOM_SUBFACADE_SYMBOLS,
    "linalg": NUMPY_LINALG_SUBFACADE_SYMBOLS,
    "fft": NUMPY_FFT_SUBFACADE_SYMBOLS,
}


# ---------------------------------------------------------------------------
# AST attribute-name denylist (instance-method reaches)
# ---------------------------------------------------------------------------

# Non-dunder attribute names rejected by ``compute._validate_ast`` when the
# dataframe facades are enabled.  These are pandas/numpy INSTANCE methods/attrs
# that reach the filesystem / network / pickle / arbitrary-expression evaluation
# / native libc and are therefore unreachable via the module facade but ARE
# reachable on objects the facade hands out.
#
# ``ctypes`` (round 4) blocks ``np.array([1]).ctypes`` — an ndarray instance
# exposes ``.ctypes`` (a ctypes bridge → native pointer / libc reach) regardless
# of the facade.  Blocking the attribute NAME at parse time kills every ``.ctypes``
# access form (also defends any other object that grows a ``.ctypes`` attribute).
BLOCKED_DATAFRAME_METHODS: frozenset[str] = frozenset(
    {
        # pandas DataFrame/Series writers (file / network / pickle).
        "to_pickle",
        "to_csv",
        "to_sql",
        "to_parquet",
        "to_feather",
        "to_hdf",
        "to_excel",
        "to_json",
        "to_xml",
        "to_stata",
        "to_clipboard",
        "to_gbq",
        "to_html",
        "to_orc",
        "to_latex",
        "to_markdown",
        # pandas arbitrary-expression execution.
        "eval",
        "query",
        # numpy file / pickle writers + readers (instance + module names).
        # Bare ``load``/``save``/``savez`` are denylisted too (belt-and-suspenders
        # vs a version where they are reachable through an internal submodule the
        # facade does not re-export); the facade already omits them at module
        # level, so this only closes the internal-path reach.
        "tofile",
        "dump",
        "dumps",
        "load",
        "loads",
        "save",
        "savez",
        "savez_compressed",
        "fromfile",
        "frombuffer",
        "fromstring",
        "fromregex",
        "memmap",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        # ndarray native-libc bridge (round 4): ``np.array([1]).ctypes`` exposes a
        # ctypes pointer object → native memory / libc reach.
        "ctypes",
        # matplotlib file readers (matplotlib is dropped from the sandbox in
        # round 4, but these names stay denylisted as belt-and-suspenders so a
        # future re-add cannot regress an IO reach without an explicit decision).
        "imread",
        "imsave",
    }
)


# ---------------------------------------------------------------------------
# Facade builders
# ---------------------------------------------------------------------------


def _build_facade(
    module_name: str,
    real_module: Any,
    symbols: tuple[str, ...],
) -> types.ModuleType:
    """Build a curated facade ``ModuleType`` re-exporting only *symbols*.

    A facade is a fresh :class:`types.ModuleType` whose attributes are copied
    from *real_module* for each name in *symbols* that the installed version
    actually provides.  Names absent from the installed version are skipped (a
    facade must work across library versions without raising).  The facade
    exposes NOTHING else — every omitted ``read_*`` / ``load`` / ``save`` entry
    point is simply not present, so ``facade.read_csv`` raises ``AttributeError``.

    SECURITY (spec §5.2, structural invariant): the facade deliberately does NOT
    stash the real backing module under any attribute, and the symbols copied in
    must themselves be non-module values (callables/types/scalars).  Whether a
    copied symbol is a live module is asserted at construction by
    :func:`_assert_no_live_module` (called from :func:`build_dataframe_modules`),
    so a regression that lists a submodule name in *symbols* fails fast.  The
    library's OWN internal lazy imports (numpy's ``from numpy._core import
    _methods``) are satisfied from ``sys.modules`` (pre-warmed via
    :data:`_NUMPY_PREWARM_SUBMODULES`), never by walking the facade — so no
    real-module handle needs to live on the facade.
    """
    facade = types.ModuleType(module_name)
    facade.__doc__ = (
        f"Curated safe facade over {module_name} for the compute sandbox. "
        "File/network/pickle entry points are omitted by design."
    )
    exported: list[str] = []
    for name in symbols:
        value = getattr(real_module, name, None)
        if value is None and not hasattr(real_module, name):
            continue
        setattr(facade, name, value)
        exported.append(name)
    # A facade-level ``__all__`` documents the surface (and bounds ``from x import *``).
    facade.__all__ = exported  # type: ignore[attr-defined]
    return facade


def build_stdlib_facade(
    module_name: str,
    real_module: Any,
    *,
    extra_denied: frozenset[str] = frozenset(),
) -> types.ModuleType:
    """Build a curated facade for an allowed stdlib (or extra) sandbox module.

    GENERIC, complete-by-construction (security review R6).  Unlike
    :func:`_build_facade` (which copies an explicit numpy/pandas allowlist), this
    introspects ``real_module`` and copies EVERY public attribute that is NOT a
    live module — so the facade preserves the module's legitimate public API
    (callables/classes/constants) while removing EVERY module handle.  The escape
    it closes: stdlib modules re-export their own imports as plain non-dunder
    attributes (``calendar.sys`` is the real ``sys``; ``fractions.operator`` is the
    real ``operator``; ``statistics.sys`` / ``datetime.sys`` / ``re.functools`` /
    ``re.enum`` / ``json.decoder`` / ``collections.abc`` …).  Reaching one of
    those by ATTRIBUTE on a pre-injected module (``calendar.sys.modules['os'].
    system('id')``) is invisible to the AST attr-name guard (a ``Subscript`` /
    non-dunder ``Attribute`` it never inspects) and is proven RCE.

    A copied attribute is EXCLUDED when ANY of:
      * ``isinstance(value, types.ModuleType)`` — the core invariant: no live
        module handle (kills ``calendar.sys`` / ``fractions.operator`` /
        ``statistics.sys`` / ``collections.abc`` / ``re.functools`` …).
      * the name starts with ``_`` — drops dunders AND single-underscore module
        handles (``collections._sys`` / ``collections._collections_abc``) and
        private internals (matches the namespace-snapshot convention).
      * the name is in *extra_denied* — the caller's denylist (e.g. the AST
        dataframe-method denylist) so a denied symbol is not even present.

    The result is a fresh :class:`types.ModuleType` whose ``__name__`` is
    *module_name*.  ``build_dataframe_modules`` / the compute sandbox run it
    through :func:`assert_no_live_module` at construction, so a future stdlib
    version that grows a new module-typed public attribute is removed
    automatically (no code change) and the invariant cannot silently regress.
    """
    facade = types.ModuleType(module_name)
    facade.__doc__ = (
        f"Curated safe facade over {module_name} for the compute sandbox. "
        "Re-exported live module handles are removed by design (no module "
        "object is reachable from sandbox code)."
    )
    exported: list[str] = []
    for name in dir(real_module):
        if name.startswith("_"):
            continue
        if name in extra_denied:
            continue
        value = getattr(real_module, name, None)
        if isinstance(value, types.ModuleType):
            continue
        setattr(facade, name, value)
        exported.append(name)
    facade.__all__ = exported  # type: ignore[attr-defined]
    return facade


def build_pandas_facade() -> types.ModuleType | None:
    """Return a safe pandas facade, or ``None`` if pandas is not installed.

    Optional dependency: a missing pandas yields ``None`` (the sandbox then
    reports ``pandas`` as unavailable), never an import error at construction.
    """
    try:
        import pandas as real_pandas
    except ImportError:
        logger.info("COMPUTE_DATAFRAME pandas=unavailable reason=not_installed")
        return None
    return _build_facade("pandas", real_pandas, PANDAS_FACADE_SYMBOLS)


def _prewarm_numpy_internals() -> None:
    """Import numpy reduction internals into ``sys.modules`` (idempotent).

    ``ndarray.mean()``/``.std()``/``.var()`` fire a RUNTIME ``__import__(
    'numpy._core._methods', fromlist=())`` from numpy's OWN bytecode every call.
    The sandbox ``__import__`` only permits such a dotted facade import when the
    submodule is ALREADY cached (it then returns the curated facade, never the
    internal module).  Warming them at facade-build time guarantees that branch
    hits regardless of numpy version / call ordering.  A bare ``import numpy``
    already loads these, so this is a defensive no-op on current numpy; failures
    are swallowed (the cache branch still falls back to the bare-import warmth).
    """
    import importlib

    for submodule in _NUMPY_PREWARM_SUBMODULES:
        try:
            importlib.import_module(submodule)
        except Exception:  # pragma: no cover - version-specific module layout
            logger.debug(
                "COMPUTE_DATAFRAME numpy_prewarm_skip submodule=%s", submodule
            )


def build_numpy_facade() -> types.ModuleType | None:
    """Return a safe numpy facade, or ``None`` if numpy is not installed.

    The facade re-exports the curated top-level :data:`NUMPY_FACADE_SYMBOLS` AND
    — crucially for the structural invariant — replaces ``random`` / ``linalg`` /
    ``fft`` with CURATED SUB-FACADES (:data:`_NUMPY_SUBFACADES`), never the REAL
    submodules.  The real submodules re-export their own imports
    (``random.np.ctypeslib`` / ``linalg.linalg.overrides.os``) — the proven R3
    live-module escapes — so handing them to sandbox code re-opens the hole.  A
    sub-facade is a fresh ``ModuleType`` with only safe pure-compute callables.

    Optional dependency (numpy ships in the framework ``search`` extra): a
    missing numpy yields ``None``, never an import error at construction.
    """
    try:
        import numpy as real_numpy
    except ImportError:
        logger.info("COMPUTE_DATAFRAME numpy=unavailable reason=not_installed")
        return None
    _prewarm_numpy_internals()
    facade = _build_facade("numpy", real_numpy, NUMPY_FACADE_SYMBOLS)
    # Replace the REAL random/linalg/fft submodules with curated sub-facades so
    # no live module object is ever reachable from ``np.random`` / ``np.linalg`` /
    # ``np.fft`` (round 4 structural invariant).
    for sub_name, sub_symbols in _NUMPY_SUBFACADES.items():
        real_sub = getattr(real_numpy, sub_name, None)
        if real_sub is None:
            continue
        sub_facade = _build_facade(f"numpy.{sub_name}", real_sub, sub_symbols)
        setattr(facade, sub_name, sub_facade)
    return facade


# ---------------------------------------------------------------------------
# Build-time structural assertion (defense-in-depth)
# ---------------------------------------------------------------------------


def _assert_no_live_module(facade: types.ModuleType) -> None:
    """Fail fast if *facade* exposes any live ``types.ModuleType`` attribute.

    Enforces the round-4 structural invariant AT CONSTRUCTION: the sandbox may
    reach ONLY curated facades + safe callables, NEVER a live module.  Walks the
    facade's public attributes and, one level deeper, each sub-facade's public
    attributes — so a regression that re-exports the REAL ``numpy.random`` (or
    leaks ``os`` onto a sub-facade) raises here at build time instead of becoming
    the next escape.  This makes regressions structural, not whack-a-mole.

    The facade object ITSELF is a ``ModuleType`` (that is the contract with the
    sandbox importer) — only its *attributes* are checked, and a curated
    sub-facade attribute is recursed into (one level) rather than rejected.
    """
    for attr_name in dir(facade):
        if attr_name.startswith("__"):
            continue
        attr_value = getattr(facade, attr_name, None)
        if not isinstance(attr_value, types.ModuleType):
            continue
        # A ModuleType attribute is allowed ONLY if it is one of our curated
        # sub-facades (built by ``_build_facade``, so dependency-free and
        # itself recursively module-free). The real numpy submodules would
        # carry ``os``/``ctypeslib``/etc. and fail the inner check.
        for sub_attr_name in dir(attr_value):
            if sub_attr_name.startswith("__"):
                continue
            sub_attr_value = getattr(attr_value, sub_attr_name, None)
            if isinstance(sub_attr_value, types.ModuleType):
                raise RuntimeError(
                    "compute sandbox facade invariant violated: "
                    f"'{facade.__name__}.{attr_name}.{sub_attr_name}' is a live "
                    "module (types.ModuleType); facades must expose only curated "
                    "callables/types, never a live module"
                )


def assert_no_live_module(facade: types.ModuleType) -> None:
    """Public alias of :func:`_assert_no_live_module` (security review R6).

    The compute sandbox runs this over EVERY module it injects into the sandbox
    namespace — not just the numpy/pandas dataframe facades — so the structural
    invariant ("no live module object, and no module-typed attribute one level
    deep, is reachable from sandbox code") is enforced AT CONSTRUCTION for the
    whole namespace.  A stdlib (or extra) module that leaks a live module handle
    therefore fails fast at tool build, never in production.
    """
    _assert_no_live_module(facade)


def build_dataframe_modules() -> dict[str, types.ModuleType]:
    """Build the ``{import-name: module}`` map for installed dataframe libs.

    Returns a mapping suitable for extending the compute sandbox import
    allowlist: curated ``pandas`` / ``numpy`` facades.  matplotlib is NOT
    included (round 4: dropped from the sandbox — its ``plt.sys`` / Figure-canvas
    pivot surface cannot be safely faceted here; in-sandbox chart rendering is
    DEFERRED pending a safe matplotlib strategy).  Only libraries that import
    successfully are included, so the result is empty when none are installed
    (graceful degradation — the default sandbox stays byte-identical).

    SECURITY (spec §5.2, round 4): each facade is run through
    :func:`_assert_no_live_module` before being returned, so a regression that
    re-leaks a live module fails fast at construction rather than silently
    becoming the next sandbox escape.
    """
    modules: dict[str, types.ModuleType] = {}
    pandas_facade = build_pandas_facade()
    if pandas_facade is not None:
        modules["pandas"] = pandas_facade
    numpy_facade = build_numpy_facade()
    if numpy_facade is not None:
        modules["numpy"] = numpy_facade
    # Defense-in-depth: assert the structural invariant on every facade.
    for facade in modules.values():
        _assert_no_live_module(facade)
    return modules

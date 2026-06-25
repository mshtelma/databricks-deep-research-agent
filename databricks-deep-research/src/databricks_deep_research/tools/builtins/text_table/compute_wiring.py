"""Inject ``table_*`` callables into a :class:`PythonComputeTool` namespace.

The 6 ``table_*`` tools each implement the :class:`ComputeCallableProvider`
Protocol — they expose a ``compute_name`` and a
``to_compute_callable(*, compute)`` factory that returns a plain Python
callable. ``inject_table_callables`` consults each provider, builds its
callable, and registers it under ``provider.compute_name`` on the hosting
``PythonComputeTool`` via the public ``inject_variable`` method.

Bindings metadata is also exposed:

- ``bindings`` — a :class:`MappingProxyType` snapshot of the registry at
  injection time. It does NOT reflect mid-turn registrations.
- ``bindings_live`` — a live read-only view that tracks subsequent
  registrations (e.g. discovery results landing during the same turn).
- ``vector_indexes`` — an optional snapshot of configured vector-search index
  metadata, injected when supplied by the caller.

The function is **idempotent**: calling it twice with the same providers
re-registers the callables on top of the previous entries.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from .budgets import Budget3D
from .compute_provider import ComputeCallableProvider
from .registry import TableBindingRegistry

__all__ = ["inject_table_callables"]


def inject_table_callables(
    *,
    compute: Any,
    providers: Iterable[ComputeCallableProvider],
    registry: TableBindingRegistry | None = None,
    vector_indexes: Mapping[str, Any] | None = None,
    expose_bindings: bool = True,
) -> list[str]:
    """Inject ``compute_name``-keyed callables onto ``compute``.

    Parameters
    ----------
    compute:
        The hosting :class:`PythonComputeTool` instance. Must expose
        ``inject_variable(name, value)``.
    providers:
        Iterable of :class:`ComputeCallableProvider` instances (the 6
        ``table_*`` tools). Order is preserved; later entries with the
        same ``compute_name`` overwrite earlier ones.
    registry:
        When provided, also exposes ``bindings`` (snapshot) and
        ``bindings_live`` (live view) on the compute namespace. Only
        applied when ``expose_bindings=True``.
    vector_indexes:
        Optional configured vector-search index metadata snapshot to expose
        under ``vector_indexes``.
    expose_bindings:
        Skip the ``bindings`` / ``bindings_live`` injection when False.

    Returns
    -------
    list[str]
        The names actually injected (in registration order).
    """
    if not hasattr(compute, "inject_variable"):
        raise TypeError(
            "compute must expose an 'inject_variable(name, value)' method; "
            f"got {type(compute).__name__}"
        )

    provider_list = list(providers)

    def _row_count(result: Any) -> int:
        if isinstance(result, list):
            return len(result)
        if isinstance(result, tuple):
            return len(result)
        if isinstance(result, Mapping):
            rows = result.get("rows")
            if isinstance(rows, (list, tuple)):
                return len(rows)
        return 0

    def _budgeted(callable_obj: Any, budget: Budget3D) -> Any:
        def _call(*args: Any, **kwargs: Any) -> Any:
            budget.tick()
            started = time.monotonic()
            result = callable_obj(*args, **kwargs)
            elapsed = time.monotonic() - started
            budget.tick(calls=0, rows=_row_count(result), wall_clock_s=elapsed)
            return result

        return _call

    def _inject_now(*, budget: Budget3D | None) -> list[str]:
        names: list[str] = []
        for provider in provider_list:
            name = provider.compute_name
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"provider {type(provider).__name__} returned an "
                    f"invalid compute_name: {name!r}"
                )
            callable_obj = provider.to_compute_callable(compute=compute)
            if budget is not None:
                callable_obj = _budgeted(callable_obj, budget)
            compute.inject_variable(name, callable_obj)
            names.append(name)

        if expose_bindings and registry is not None:
            compute.inject_variable("bindings", registry.metadata_snapshot())
            compute.inject_variable("bindings_live", registry.metadata_view())

        if vector_indexes is not None:
            compute.inject_variable("vector_indexes", dict(vector_indexes))

        return names

    if hasattr(compute, "set_before_execute_hook"):
        def _refresh(host_compute: Any) -> None:
            del host_compute
            _inject_now(budget=Budget3D())

        compute.set_before_execute_hook("text_table_callables", _refresh)

    injected: list[str] = []
    injected.extend(_inject_now(budget=None))
    return injected

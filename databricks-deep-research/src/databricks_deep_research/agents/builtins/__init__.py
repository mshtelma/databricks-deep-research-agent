"""Builtin agent subtypes.

Importing this package registers all builtin subtypes with the
builtin registry so they are available to the harness automatically.
"""

import databricks_deep_research.agents.builtins.background  # noqa: F401

# Import each builtin module to trigger registration
import databricks_deep_research.agents.builtins.coordinator  # noqa: F401
import databricks_deep_research.agents.builtins.custom  # noqa: F401
import databricks_deep_research.agents.builtins.planner  # noqa: F401
import databricks_deep_research.agents.builtins.reflector  # noqa: F401
import databricks_deep_research.agents.builtins.researcher  # noqa: F401
import databricks_deep_research.agents.builtins.synthesizer  # noqa: F401
from databricks_deep_research.agents.builtins.registry import (  # noqa: F401
    get_builtin,
    list_builtins,
    register_builtin,
)

__all__ = ["get_builtin", "list_builtins", "register_builtin"]

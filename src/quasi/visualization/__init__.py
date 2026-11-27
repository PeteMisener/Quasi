"""Publication-ready matplotlib figure renderers for Quasi estimators (lazily imported)."""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "render_discontinuity_figure": "quasi.visualization.discontinuity",
    "render_event_study_figure": "quasi.visualization.event_study",
    "render_its_figure": "quasi.visualization.its_plots",
    "render_love_figure": "quasi.visualization.love_plot",
    "render_synth_figure": "quasi.visualization.synth_plots",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        try:
            mod = importlib.import_module(_EXPORTS[name])
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("matplotlib"):
                raise ImportError(
                    "Plotting requires matplotlib. Install with `pip install 'quasi-qed[plot]'`."
                ) from exc
            raise
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

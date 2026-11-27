"""
Multi-Model Comparison Table Data Model and compare_models() constructor.
Aligns heterogeneous QED results and orchestrates document serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np

from quasi.base import QEDResult


def _is_finite(val: Any) -> bool:
    return isinstance(val, (int, float, np.number)) and not isinstance(val, bool) and math.isfinite(float(val))


@dataclass
class TableFormattingOptions:
    """Styling and layout options for multi-model comparison tables."""
    title: Optional[str] = None
    notes: Optional[str] = None
    se_type: Literal["parens", "brackets", "ci"] = "parens"
    stars: Dict[float, str] = field(default_factory=lambda: {0.10: "*", 0.05: "**", 0.01: "***"})
    digits: int = 3
    include_specs: bool = True
    include_diagnostics: bool = True
    include_sample_stats: bool = True
    style: Literal["aer", "nature", "compact"] = "aer"


class ModelComparisonTable:
    """
    Comparison table container aggregating multiple QEDResult objects.
    Renders to LaTeX, Typst, Markdown, HTML, and terminal ANSI tables.
    """

    def __init__(
        self,
        models: Union[List[QEDResult], Dict[str, QEDResult]],
        options: Optional[TableFormattingOptions] = None,
    ) -> None:
        if isinstance(models, list):
            if not models:
                raise ValueError("compare_models requires at least one QEDResult")
            self.models: Dict[str, QEDResult] = {
                f"Model {i + 1}": m for i, m in enumerate(models)
            }
        else:
            if not models:
                raise ValueError("compare_models requires at least one QEDResult")
            self.models = models

        base_opts = options or TableFormattingOptions()
        if base_opts.style == "nature":
            # Copy options before overriding so caller's TableFormattingOptions is never mutated in place
            self.options = replace(base_opts, stars={}, se_type="brackets")
        else:
            self.options = replace(base_opts, stars=dict(base_opts.stars))

        self._matrix = self._build_comparison_matrix()

    def _build_comparison_matrix(self) -> Dict[str, Any]:
        """Extracts aligned metrics across disparate model architectures."""
        matrix: Dict[str, Any] = {
            "estimates": {},
            "specs": {},
            "diagnostics": {},
            "sample_stats": {},
            "confidence_levels": [],
        }

        # Collect ordered union of all diagnostic names reported across models.
        all_diag_names: List[str] = []
        raw_metas: Dict[str, Dict[str, Any]] = {}
        for name, res in self.models.items():
            meta = res.to_metadata_dict()
            raw_metas[name] = meta
            for d in meta.get("diagnostics", []):
                dname = d.get("name")
                if dname and dname not in all_diag_names:
                    all_diag_names.append(dname)

        for name, res in self.models.items():
            meta = raw_metas[name]
            pm = meta["primary_metric"]
            p_val = pm.get("p_value")
            est = pm.get("estimate")
            se = pm.get("se")
            ci = pm.get("ci")
            conf_level = float(pm.get("confidence_level", res.confidence_level))
            matrix["confidence_levels"].append(conf_level)

            # 1. Panel A: Format point estimates & stars (point-only safe)
            star_symbol = ""
            if _is_finite(p_val):
                for thresh in sorted(self.options.stars.keys()):
                    if float(p_val) <= thresh:
                        star_symbol = self.options.stars[thresh]
                        break

            est_str = f"{float(est):+.{self.options.digits}f}{star_symbol}" if _is_finite(est) else "N/A"
            if self.options.se_type == "parens":
                se_or_ci_str = f"({float(se):.{self.options.digits}f})" if _is_finite(se) else "(N/A)"
            else:
                if (
                    isinstance(ci, (tuple, list))
                    and len(ci) == 2
                    and _is_finite(ci[0])
                    and _is_finite(ci[1])
                ):
                    se_or_ci_str = f"[{float(ci[0]):.{self.options.digits}f}, {float(ci[1]):.{self.options.digits}f}]"
                else:
                    se_or_ci_str = "[N/A]"

            matrix["estimates"][name] = {
                "est": est_str,
                "se_or_ci": se_or_ci_str,
                "raw_est": est,
                "raw_se": se,
                "raw_p": p_val,
                "confidence_level": conf_level,
            }

            # 2. Panel B: Specifications
            matrix["specs"][name] = {
                "Model Class": meta.get("model_name", "QED"),
                "Estimand": meta.get("estimand", "ATT"),
                "Outcome Variable": meta.get("outcome_name", "Y"),
                "Inference": meta.get("inference_method", "unverified"),
            }

            # 3. Panel C: Live QED Identification & Assumption Diagnostics
            diag_map = {d["name"]: d for d in meta.get("diagnostics", [])}
            diag_row: Dict[str, str] = {
                "Diagnostic Status": str(meta.get("diagnostic_status", "not_run")),
            }
            for dname in all_diag_names:
                entry = diag_map.get(dname)
                if entry is None:
                    diag_row[dname] = "-"
                elif _is_finite(entry.get("p_value")):
                    diag_row[dname] = f"p={float(entry['p_value']):.3f} [{entry['status']}]"
                elif _is_finite(entry.get("statistic_value")):
                    diag_row[dname] = f"{float(entry['statistic_value']):.3f} [{entry['status']}]"
                else:
                    diag_row[dname] = f"[{entry.get('status', 'WARN')}]"
            matrix["diagnostics"][name] = diag_row

            # 4. Panel D: Sample Statistics
            s_metrics = meta.get("sample_metrics", {})
            matrix["sample_stats"][name] = {
                "Observations (N)": f"{s_metrics.get('n_obs', res.n_obs):,}",
                "Treated Units": f"{s_metrics.get('n_treated', res.n_treated):,}",
            }

        return matrix

    def to_latex(self, file_path: Optional[str] = None, booktabs: bool = True) -> str:
        """Exports comparison table to publication-grade LaTeX."""
        from quasi.reporting.renderers import LatexRenderer
        content = LatexRenderer(self._matrix, self.options).render(booktabs=booktabs)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def to_markdown(self, file_path: Optional[str] = None) -> str:
        """Exports comparison table to GitHub-Flavored Markdown."""
        from quasi.reporting.renderers import MarkdownRenderer
        content = MarkdownRenderer(self._matrix, self.options).render()
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def to_typst(self, file_path: Optional[str] = None) -> str:
        """Exports comparison table to modern Typst table format."""
        from quasi.reporting.renderers import TypstRenderer
        content = TypstRenderer(self._matrix, self.options).render()
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def to_html(self, file_path: Optional[str] = None) -> str:
        """Exports comparison table to styled HTML."""
        from quasi.reporting.renderers import HtmlRenderer
        content = HtmlRenderer(self._matrix, self.options).render()
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def print_terminal(self) -> None:
        """Displays formatted ANSI comparison table in the console using rich."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=self.options.title or "quasi: Multi-Model Specification Comparison")
        table.add_column("Specification / Metric", style="cyan", no_wrap=True)

        model_names = list(self.models.keys())
        for i, name in enumerate(model_names):
            table.add_column(f"({i + 1}) {name}", justify="center")

        # Estimates
        est_row = ["Treatment Effect (ATT/LATE)"] + [self._matrix["estimates"][m]["est"] for m in model_names]
        se_row = ["Std. Error / (CI)"] + [self._matrix["estimates"][m]["se_or_ci"] for m in model_names]
        table.add_row(*est_row)
        table.add_row(*se_row)
        table.add_section()

        # Model Class
        spec_row = ["Model Architecture"] + [self._matrix["specs"][m]["Model Class"] for m in model_names]
        table.add_row(*spec_row)
        table.add_section()

        # Active Diagnostics
        diag_keys = list(next(iter(self._matrix["diagnostics"].values())).keys())
        for key in diag_keys:
            vals = [self._matrix["diagnostics"][m].get(key, "-") for m in model_names]
            if any(v != "-" for v in vals):
                table.add_row(key, *vals)
        table.add_section()

        # Sample Stats
        stat_keys = list(next(iter(self._matrix["sample_stats"].values())).keys())
        for key in stat_keys:
            table.add_row(key, *[self._matrix["sample_stats"][m][key] for m in model_names])

        console.print(table)

    def _repr_html_(self) -> str:
        """Interactive rich HTML representation for Jupyter notebooks."""
        return self.to_html()


def compare_models(
    models: Union[List[QEDResult], Dict[str, QEDResult]],
    title: Optional[str] = None,
    notes: Optional[str] = None,
    se_type: Literal["parens", "brackets", "ci"] = "parens",
    style: Literal["aer", "nature", "compact"] = "aer",
    digits: int = 3,
    include_specs: bool = True,
    include_diagnostics: bool = True,
    include_sample_stats: bool = True,
) -> ModelComparisonTable:
    """
    Constructs a multi-model comparative table across diverse quasi-experimental estimates.

    Parameters:
        models: List or dictionary of fitted QEDResult objects.
        title: Optional table caption or title.
        notes: Optional custom footer notes.
        se_type: 'parens' for (SE) or 'brackets' for [CI].
        style: 'aer' (American Economic Review standard), 'nature' (Nature portfolio standard), or 'compact'.
        digits: Floating point rounding precision.
    """
    options = TableFormattingOptions(
        title=title,
        notes=notes,
        se_type=se_type,
        style=style,
        digits=digits,
        include_specs=include_specs,
        include_diagnostics=include_diagnostics,
        include_sample_stats=include_sample_stats,
    )
    return ModelComparisonTable(models=models, options=options)

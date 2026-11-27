"""
Specialized publication-ready document renderers for multi-model comparison tables.
Supports LaTeX (AER, Nature), Typst, GitHub-Flavored Markdown, and HTML.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List
from quasi.reporting.table import TableFormattingOptions


def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters in text strings."""
    return (
        text.replace("\\", "\\textbackslash ")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde ")
        .replace("^", "\\textasciicircum ")
    )


def escape_typst(text: str) -> str:
    """Escapes Typst markup delimiters inside content blocks."""
    return (
        text.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("*", "\\*")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("_", "\\_")
    )


def _ci_label(data: Dict[str, Any]) -> str:
    levels = sorted({round(float(x) * 100.0, 2) for x in data.get("confidence_levels", [0.95])})
    if len(levels) == 1:
        lvl = levels[0]
        return f"{int(lvl)}%" if lvl.is_integer() else f"{lvl:g}%"
    return "model-level"


class LatexRenderer:
    """Renders ModelComparisonTable into publication-grade LaTeX with booktabs."""

    def __init__(self, data: Dict[str, Any], options: TableFormattingOptions) -> None:
        self.data = data
        self.options = options
        self.model_names = list(self.data["estimates"].keys())
        self.n_models = len(self.model_names)

    def render(self, booktabs: bool = True) -> str:
        col_spec = "l" + ("c" * self.n_models)
        lines: List[str] = []

        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        if self.options.title:
            lines.append(f"\\caption{{{escape_latex(self.options.title)}}}")
        lines.append("\\label{tab:quasi_comparison}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        # Top rule
        lines.append("\\toprule" if booktabs else "\\hline\\hline")

        # Header row with model names
        headers = [""] + [f"\\multicolumn{{1}}{{c}}{{{escape_latex(name)}}}" for name in self.model_names]
        lines.append(" & ".join(headers) + " \\\\")

        # Model column numbers: (1), (2), ...
        col_nums = [""] + [f"({i+1})" for i in range(self.n_models)]
        lines.append(" & ".join(col_nums) + " \\\\")
        lines.append("\\midrule" if booktabs else "\\hline")

        # Panel A: Causal Estimates
        span = self.n_models + 1
        lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{Panel A: Causal Estimates}}}} \\\\")
        
        # Point estimates row
        est_row = ["Treatment Effect"] + [self.data["estimates"][m]["est"] for m in self.model_names]
        lines.append(" & ".join(est_row) + " \\\\")
        
        # Uncertainty row (SE in parens or CI in brackets)
        se_row = [""] + [self.data["estimates"][m]["se_or_ci"] for m in self.model_names]
        lines.append(" & ".join(se_row) + " \\\\")
        lines.append("\\midrule" if booktabs else "\\hline")

        # Panel B: Model Specifications
        if self.options.include_specs and self.data["specs"]:
            lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{Panel B: Model Specification}}}} \\\\")
            spec_keys = list(next(iter(self.data["specs"].values())).keys())
            for key in spec_keys:
                row = [escape_latex(key)] + [escape_latex(str(self.data["specs"][m].get(key, "-"))) for m in self.model_names]
                lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule" if booktabs else "\\hline")

        # Panel C: QED Identification Diagnostics
        if self.options.include_diagnostics and self.data["diagnostics"]:
            lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{Panel C: Identification Diagnostics}}}} \\\\")
            diag_keys = list(next(iter(self.data["diagnostics"].values())).keys())
            for key in diag_keys:
                # Check if at least one model reported this diagnostic
                vals = [self.data["diagnostics"][m].get(key, "-") for m in self.model_names]
                if any(v != "-" for v in vals):
                    row = [escape_latex(key)] + [escape_latex(str(v)) for v in vals]
                    lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule" if booktabs else "\\hline")

        # Panel D: Sample Statistics
        if self.options.include_sample_stats and self.data["sample_stats"]:
            lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{Panel D: Sample Statistics}}}} \\\\")
            stat_keys = list(next(iter(self.data["sample_stats"].values())).keys())
            for key in stat_keys:
                row = [escape_latex(key)] + [escape_latex(str(self.data["sample_stats"][m].get(key, "-"))) for m in self.model_names]
                lines.append(" & ".join(row) + " \\\\")

        # Bottom rule
        lines.append("\\bottomrule" if booktabs else "\\hline\\hline")
        lines.append("\\end{tabular}")

        # Tablenotes / captions
        ci_lbl = _ci_label(self.data).replace("%", "\\%")
        if self.options.notes:
            notes = escape_latex(self.options.notes)
        else:
            star_txt = "* $p<0.10$, ** $p<0.05$, *** $p<0.01$. " if self.options.stars else ""
            unc_txt = (
                "Standard errors reported in parentheses."
                if self.options.se_type == "parens"
                else f"Point estimates with {ci_lbl} confidence intervals reported in brackets."
            )
            notes = star_txt + unc_txt
        lines.append("\\begin{tablenotes}[flushleft]")
        lines.append(f"\\small\\item \\textit{{Notes:}} {notes}")
        lines.append("\\end{tablenotes}")
        lines.append("\\end{table}")

        return "\n".join(lines)


class MarkdownRenderer:
    """Renders ModelComparisonTable into clean GitHub-Flavored Markdown."""

    def __init__(self, data: Dict[str, Any], options: TableFormattingOptions) -> None:
        self.data = data
        self.options = options
        self.model_names = list(self.data["estimates"].keys())

    def render(self) -> str:
        lines: List[str] = []
        if self.options.title:
            lines.append(f"### {self.options.title}\n")

        headers = [""] + [f"({i+1}) {m}" for i, m in enumerate(self.model_names)]
        align = [":---"] + [":---:" for _ in self.model_names]

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(align) + " |")

        # Panel A
        lines.append(f"| **Panel A: Causal Estimates** |" + " |" * len(self.model_names))
        est_row = ["Treatment Effect"] + [self.data["estimates"][m]["est"] for m in self.model_names]
        lines.append("| " + " | ".join(est_row) + " |")
        se_row = [""] + [self.data["estimates"][m]["se_or_ci"] for m in self.model_names]
        lines.append("| " + " | ".join(se_row) + " |")

        # Panel B
        if self.options.include_specs and self.data["specs"]:
            lines.append(f"| **Panel B: Model Specification** |" + " |" * len(self.model_names))
            spec_keys = list(next(iter(self.data["specs"].values())).keys())
            for key in spec_keys:
                row = [key] + [str(self.data["specs"][m].get(key, "-")) for m in self.model_names]
                lines.append("| " + " | ".join(row) + " |")

        # Panel C
        if self.options.include_diagnostics and self.data["diagnostics"]:
            lines.append(f"| **Panel C: Identification Diagnostics** |" + " |" * len(self.model_names))
            diag_keys = list(next(iter(self.data["diagnostics"].values())).keys())
            for key in diag_keys:
                vals = [self.data["diagnostics"][m].get(key, "-") for m in self.model_names]
                if any(v != "-" for v in vals):
                    row = [key] + [str(v) for v in vals]
                    lines.append("| " + " | ".join(row) + " |")

        # Panel D
        if self.options.include_sample_stats and self.data["sample_stats"]:
            lines.append(f"| **Panel D: Sample Statistics** |" + " |" * len(self.model_names))
            stat_keys = list(next(iter(self.data["sample_stats"].values())).keys())
            for key in stat_keys:
                row = [key] + [str(self.data["sample_stats"][m].get(key, "-")) for m in self.model_names]
                lines.append("| " + " | ".join(row) + " |")

        ci_lbl = _ci_label(self.data)
        if self.options.notes:
            notes = self.options.notes
        else:
            star_txt = "* p < 0.10, ** p < 0.05, *** p < 0.01. " if self.options.stars else ""
            unc_txt = (
                "Standard errors in parentheses."
                if self.options.se_type == "parens"
                else f"Point estimates reported with {ci_lbl} confidence intervals in brackets."
            )
            notes = f"*Notes:* {star_txt}{unc_txt}"
        lines.append(f"\n{notes}")
        return "\n".join(lines)


class TypstRenderer:
    """Renders ModelComparisonTable into modern Typst document markup."""

    def __init__(self, data: Dict[str, Any], options: TableFormattingOptions) -> None:
        self.data = data
        self.options = options
        self.model_names = list(self.data["estimates"].keys())
        self.n_models = len(self.model_names)

    def render(self) -> str:
        lines: List[str] = []
        lines.append("#figure(")
        lines.append("  table(")
        
        # Columns spec: 2fr for labels, 1.2fr per model
        col_widths = ["2.2fr"] + ["1.2fr"] * self.n_models
        lines.append(f"    columns: ({', '.join(col_widths)}),")
        lines.append("    align: (left, " + ", ".join(["center"] * self.n_models) + "),")
        lines.append("    stroke: none,")
        lines.append("    table.hline(stroke: 1.5pt),")

        # Header
        hdr = ["[]"] + [f"[*({i+1}) {escape_typst(m)}*]" for i, m in enumerate(self.model_names)]
        lines.append("    " + ", ".join(hdr) + ",")
        lines.append("    table.hline(stroke: 0.8pt),")

        span = self.n_models + 1
        # Panel A
        lines.append(f"    table.cell(colspan: {span})[*Panel A: Causal Estimates*],")
        est_row = ["[Treatment Effect]"] + [f"[{escape_typst(self.data['estimates'][m]['est'])}]" for m in self.model_names]
        lines.append("    " + ", ".join(est_row) + ",")
        se_row = ["[]"] + [f"[{escape_typst(self.data['estimates'][m]['se_or_ci'])}]" for m in self.model_names]
        lines.append("    " + ", ".join(se_row) + ",")
        lines.append("    table.hline(stroke: 0.5pt),")

        # Panel B
        if self.options.include_specs and self.data["specs"]:
            lines.append(f"    table.cell(colspan: {span})[*Panel B: Model Specification*],")
            spec_keys = list(next(iter(self.data["specs"].values())).keys())
            for key in spec_keys:
                row = [f"[{escape_typst(key)}]"] + [f"[{escape_typst(str(self.data['specs'][m].get(key, '-')))}]" for m in self.model_names]
                lines.append("    " + ", ".join(row) + ",")
            lines.append("    table.hline(stroke: 0.5pt),")

        # Panel C
        if self.options.include_diagnostics and self.data["diagnostics"]:
            lines.append(f"    table.cell(colspan: {span})[*Panel C: Identification Diagnostics*],")
            diag_keys = list(next(iter(self.data["diagnostics"].values())).keys())
            for key in diag_keys:
                vals = [self.data["diagnostics"][m].get(key, "-") for m in self.model_names]
                if any(v != "-" for v in vals):
                    row = [f"[{escape_typst(key)}]"] + [f"[{escape_typst(str(v))}]" for v in vals]
                    lines.append("    " + ", ".join(row) + ",")
            lines.append("    table.hline(stroke: 0.5pt),")

        # Panel D
        if self.options.include_sample_stats and self.data["sample_stats"]:
            lines.append(f"    table.cell(colspan: {span})[*Panel D: Sample Statistics*],")
            stat_keys = list(next(iter(self.data["sample_stats"].values())).keys())
            for key in stat_keys:
                row = [f"[{escape_typst(key)}]"] + [f"[{escape_typst(str(self.data['sample_stats'][m].get(key, '-')))}]" for m in self.model_names]
                lines.append("    " + ", ".join(row) + ",")

        lines.append("    table.hline(stroke: 1.5pt),")
        lines.append("  ),")
        if self.options.title:
            lines.append(f"  caption: [{escape_typst(self.options.title)}],")
        lines.append(")")
        return "\n".join(lines)


class HtmlRenderer:
    """Renders ModelComparisonTable into styled HTML for Jupyter Notebooks."""

    def __init__(self, data: Dict[str, Any], options: TableFormattingOptions) -> None:
        self.data = data
        self.options = options
        self.model_names = list(self.data["estimates"].keys())
        self.n_models = len(self.model_names)

    def render(self) -> str:
        span = self.n_models + 1
        lines: List[str] = []
        lines.append('<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 16px 0;">')
        if self.options.title:
            lines.append(f'<h3 style="color: #1B2631; margin-bottom: 8px;">{html.escape(self.options.title)}</h3>')

        lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12.5px; text-align: center;">')
        lines.append('<thead>')
        lines.append('<tr style="border-top: 2px solid #2C3E50; border-bottom: 1.5px solid #2C3E50;">')
        lines.append('<th style="text-align: left; padding: 8px;"></th>')
        for i, m in enumerate(self.model_names):
            lines.append(f'<th style="padding: 8px; font-weight: 700;">({i+1})<br>{html.escape(m)}</th>')
        lines.append('</tr>')
        lines.append('</thead>')
        lines.append('<tbody>')

        # Panel A
        lines.append(f'<tr style="background: #F8F9F9; font-weight: 700; text-align: left;"><td colspan="{span}" style="padding: 6px 8px;">Panel A: Causal Estimates</td></tr>')
        lines.append('<tr style="border-top: 1px solid #EAECEE;">')
        lines.append('<td style="text-align: left; padding: 6px 8px; font-weight: 600;">Treatment Effect</td>')
        for m in self.model_names:
            lines.append(f'<td style="padding: 6px 8px; font-weight: 700; color: #1B4F72;">{self.data["estimates"][m]["est"]}</td>')
        lines.append('</tr>')
        lines.append('<tr>')
        lines.append('<td style="text-align: left; padding: 2px 8px;"></td>')
        for m in self.model_names:
            lines.append(f'<td style="padding: 2px 8px; color: #566573;">{self.data["estimates"][m]["se_or_ci"]}</td>')
        lines.append('</tr>')

        # Panel B
        if self.options.include_specs and self.data["specs"]:
            lines.append(f'<tr style="background: #F8F9F9; font-weight: 700; text-align: left; border-top: 1.5px solid #BDC3C7;"><td colspan="{span}" style="padding: 6px 8px;">Panel B: Model Specification</td></tr>')
            spec_keys = list(next(iter(self.data["specs"].values())).keys())
            for key in spec_keys:
                lines.append('<tr style="border-top: 1px solid #EAECEE;">')
                lines.append(f'<td style="text-align: left; padding: 6px 8px;">{html.escape(key)}</td>')
                for m in self.model_names:
                    lines.append(f'<td style="padding: 6px 8px;">{html.escape(str(self.data["specs"][m].get(key, "-")))}</td>')
                lines.append('</tr>')

        # Panel C
        if self.options.include_diagnostics and self.data["diagnostics"]:
            lines.append(f'<tr style="background: #F8F9F9; font-weight: 700; text-align: left; border-top: 1.5px solid #BDC3C7;"><td colspan="{span}" style="padding: 6px 8px;">Panel C: Identification Diagnostics</td></tr>')
            diag_keys = list(next(iter(self.data["diagnostics"].values())).keys())
            for key in diag_keys:
                vals = [self.data["diagnostics"][m].get(key, "-") for m in self.model_names]
                if any(v != "-" for v in vals):
                    lines.append('<tr style="border-top: 1px solid #EAECEE;">')
                    lines.append(f'<td style="text-align: left; padding: 6px 8px;">{html.escape(key)}</td>')
                    for v in vals:
                        lines.append(f'<td style="padding: 6px 8px;">{html.escape(str(v))}</td>')
                    lines.append('</tr>')

        # Panel D
        if self.options.include_sample_stats and self.data["sample_stats"]:
            lines.append(f'<tr style="background: #F8F9F9; font-weight: 700; text-align: left; border-top: 1.5px solid #BDC3C7;"><td colspan="{span}" style="padding: 6px 8px;">Panel D: Sample Statistics</td></tr>')
            stat_keys = list(next(iter(self.data["sample_stats"].values())).keys())
            for key in stat_keys:
                lines.append('<tr style="border-top: 1px solid #EAECEE;">')
                lines.append(f'<td style="text-align: left; padding: 6px 8px;">{html.escape(key)}</td>')
                for m in self.model_names:
                    lines.append(f'<td style="padding: 6px 8px;">{html.escape(str(self.data["sample_stats"][m].get(key, "-")))}</td>')
                lines.append('</tr>')

        lines.append('</tbody>')
        lines.append('<tfoot>')
        lines.append(f'<tr style="border-top: 2px solid #2C3E50;"><td colspan="{span}" style="text-align: left; padding: 8px; font-size: 11px; color: #5D6D7E;">')
        ci_lbl = _ci_label(self.data)
        notes = self.options.notes or (
            "* p < 0.10, ** p < 0.05, *** p < 0.01. Standard errors reported in parentheses."
            if self.options.style == "aer"
            else f"Values denote point estimates with {ci_lbl} confidence intervals reported in brackets."
        )
        lines.append(f'<em>Notes:</em> {html.escape(notes)}</td></tr>')
        lines.append('</tfoot>')
        lines.append('</table>')
        lines.append('</div>')
        return "\n".join(lines)
"""Reporting and publication export module for quasi."""

from __future__ import annotations

from quasi.reporting.html_card import generate_result_html
from quasi.reporting.provenance import generate_bibtex, generate_deterministic_explanation
from quasi.reporting.renderers import HtmlRenderer, LatexRenderer, MarkdownRenderer, TypstRenderer
from quasi.reporting.table import (
    ModelComparisonTable,
    TableFormattingOptions,
    compare_models,
)

__all__ = [
    "compare_models",
    "ModelComparisonTable",
    "TableFormattingOptions",
    "LatexRenderer",
    "MarkdownRenderer",
    "TypstRenderer",
    "HtmlRenderer",
    "generate_result_html",
    "generate_bibtex",
    "generate_deterministic_explanation",
]
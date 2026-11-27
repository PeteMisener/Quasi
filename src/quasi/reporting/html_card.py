"""Interactive HTML card rendering for Jupyter Notebooks."""

from __future__ import annotations

import html
import math
from typing import Any


def _finite(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def generate_result_html(result: Any) -> str:
    """Generates a zero-dependency, CSS-isolated interactive HTML card for Jupyter."""
    est = result.estimate
    se = result.se
    p_val = result.p_value
    ci_low, ci_high = result.ci_lower, result.ci_upper
    has_se = _finite(se)
    has_p = _finite(p_val)
    has_ci = _finite(ci_low) and _finite(ci_high)

    se_display = f"{float(se):.4f}" if has_se else "N/A"
    p_display = f"{float(p_val):.4f}" if has_p else "N/A"
    p_color = ("#117A65" if float(p_val) < (1.0 - result.confidence_level) else "#943126") if has_p else "#5D6D7E"
    ci_display = f"[{float(ci_low):.3f}, {float(ci_high):.3f}]" if has_ci else "N/A"

    outcome = html.escape(result.outcome_name)
    model_name = html.escape(result.model_name)
    estimand = html.escape(result.estimand)
    # No combination of diagnostic results can certify causal identification.
    diagnostic_label = {
        "not_run": "NO DIAGNOSTICS RUN",
        "failed": "DIAGNOSTIC FAILURE",
        "warnings": "DIAGNOSTIC WARNINGS",
        "passed_checks_only": "CHECKS PASSED; IDENTIFICATION UNVERIFIED",
    }[result.diagnostic_status]
    status_badge = (
        '<span style="background: #FFF3CD; color: #765300; padding: 3px 8px; '
        'border-radius: 12px; font-weight: 600; font-size: 12px;">'
        f"{diagnostic_label}</span>"
    )

    diag_rows = ""
    for test in result.diagnostics.values():
        color = (
            "#117A65"
            if test.status.value == "PASS"
            else ("#B7950B" if test.status.value == "WARN" else "#943126")
        )
        stat_str = f"{html.escape(test.statistic_name)} = {test.statistic_value:.3f}" if _finite(test.statistic_value) else html.escape(test.statistic_name)
        p_cell = f"{test.p_value:.3f}" if _finite(test.p_value) else "-"
        diag_rows += f"""
        <tr style="border-bottom: 1px solid #EAECEE;">
            <td style="padding: 8px; font-weight: 500;">{html.escape(test.name)}</td>
            <td style="padding: 8px; text-align: right;">{stat_str}</td>
            <td style="padding: 8px; text-align: right;">{p_cell}</td>
            <td style="padding: 8px; text-align: center; color: {color}; font-weight: 700;">{test.status.value}</td>
            <td style="padding: 8px; font-size: 12px; color: #5D6D7E;">{html.escape(test.message)}</td>
        </tr>
        """

    if not diag_rows:
        diag_rows = '<tr><td colspan="5" style="padding: 8px;">No diagnostics executed; identifying assumptions not verified.</td></tr>'
    methods_p = html.escape(result.methods_text())
    bibtex_p = html.escape(result.bibtex())

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 820px; border: 1px solid #D5D8DC; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin: 12px 0;">
        <div style="background: #FFF3CD; color: #765300; padding: 10px 18px; font-size: 12px;">EXPERIMENTAL ESTIMATOR — reference statistical validation pending. Identifying assumptions are not verified by these diagnostics.</div>
        <div style="background: #F8F9F9; padding: 14px 18px; border-bottom: 1px solid #D5D8DC; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 16px; font-weight: 700; color: #1B2631;">quasi: {model_name}</span>
                <span style="font-size: 13px; color: #5D6D7E; margin-left: 8px;">(Outcome: <strong>{outcome}</strong>)</span>
            </div>
            {status_badge}
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); background: #FFFFFF; border-bottom: 1px solid #EAECEE; text-align: center;">
            <div style="padding: 12px; border-right: 1px solid #EAECEE;">
                <div style="font-size: 11px; font-weight: 600; color: #7F8C8D; text-transform: uppercase;">Estimate ({estimand})</div>
                <div style="font-size: 18px; font-weight: 700; color: #1B4F72; margin-top: 4px;">{est:+.4f}</div>
            </div>
            <div style="padding: 12px; border-right: 1px solid #EAECEE;">
                <div style="font-size: 11px; font-weight: 600; color: #7F8C8D; text-transform: uppercase;">Std. Error</div>
                <div style="font-size: 18px; font-weight: 600; color: #2C3E50; margin-top: 4px;">{se_display}</div>
            </div>
            <div style="padding: 12px; border-right: 1px solid #EAECEE;">
                <div style="font-size: 11px; font-weight: 600; color: #7F8C8D; text-transform: uppercase;">p-value</div>
                <div style="font-size: 18px; font-weight: 600; color: {p_color}; margin-top: 4px;">{p_display}</div>
            </div>
            <div style="padding: 12px;">
                <div style="font-size: 11px; font-weight: 600; color: #7F8C8D; text-transform: uppercase;">{100*result.confidence_level:g}% Conf. Int.</div>
                <div style="font-size: 15px; font-weight: 600; color: #2C3E50; margin-top: 6px;">{ci_display}</div>
            </div>
        </div>

        <div style="padding: 16px; background: #FFFFFF;">
            <div style="font-size: 13px; line-height: 1.5; color: #2C3E50; margin-bottom: 14px; padding: 10px 14px; background: #EBF5FB; border-left: 4px solid #2980B9; border-radius: 2px;">
                <strong>Key Takeaway:</strong> {html.escape(result.explain(audience="stakeholder", engine="rule_based"))}
            </div>

            <h4 style="margin: 12px 0 8px 0; font-size: 13px; text-transform: uppercase; color: #566573; letter-spacing: 0.5px;">Identification & Diagnostics</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;">
                <thead>
                    <tr style="background: #F4F6F6; color: #566573;">
                        <th style="padding: 6px 8px;">Assumption / Gate</th>
                        <th style="padding: 6px 8px; text-align: right;">Metric</th>
                        <th style="padding: 6px 8px; text-align: right;">p-value</th>
                        <th style="padding: 6px 8px; text-align: center;">Status</th>
                        <th style="padding: 6px 8px;">Diagnostic Message</th>
                    </tr>
                </thead>
                <tbody>
                    {diag_rows}
                </tbody>
            </table>

            <details style="margin-top: 14px;">
                <summary style="font-size: 12px; font-weight: 600; color: #2980B9; cursor: pointer;">Methods Draft & Citation</summary>
                <div style="margin-top: 8px; padding: 10px; background: #F8F9F9; border-radius: 4px; font-size: 12px; color: #2C3E50;">
                    <p style="margin: 0 0 8px 0;"><strong>Draft Methodology:</strong> {methods_p}</p>
                    <pre style="background: #EAEDED; padding: 8px; border-radius: 4px; overflow-x: auto; margin: 0; font-size: 11px;">{bibtex_p}</pre>
                </div>
            </details>
        </div>
    </div>
    """

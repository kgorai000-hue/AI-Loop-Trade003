from __future__ import annotations

from src.core.types import SymbolStatsReport


def generate_strategy_diagnostic_report(
    strategy_name: str,
    period: str,
    research_reports: list[SymbolStatsReport],
    *,
    return_pct: float | None = None,
    benchmark_return_pct: float | None = None,
    max_drawdown_pct: float | None = None,
    sharpe: float | None = None,
) -> str:
    """Template-based diagnostic report (LLM assists structure, numbers from code)."""
    lines = [
        f"# {strategy_name} Diagnostic Report",
        f"Period: {period}",
        "",
        "## Summary",
    ]

    if return_pct is not None and benchmark_return_pct is not None:
        gap = return_pct - benchmark_return_pct
        lines.append(
            f"Strategy return {return_pct:+.1%} vs benchmark {benchmark_return_pct:+.1%} "
            f"(gap {gap:+.1%})."
        )
    if max_drawdown_pct is not None:
        lines.append(f"Max drawdown: {max_drawdown_pct:.1%}.")
    if sharpe is not None:
        lines.append(f"Sharpe ratio: {sharpe:.2f}.")

    lines.extend(["", "## Symbol Diagnostics (computed, not LLM-generated)"])
    for report in research_reports[:8]:
        rsi_text = f"{report.rsi:.1f}" if report.rsi is not None else "n/a"
        lines.append(
            f"- {report.symbol}: ann_return={report.annualized_return:.2%}, "
            f"vol={report.annualized_volatility:.2%}, regime={report.regime.value}, "
            f"RSI={rsi_text}"
        )

    high_vol = [r for r in research_reports if r.annualized_volatility >= 0.30]
    if high_vol:
        lines.extend(["", "## Risk Flags"])
        for report in high_vol[:5]:
            lines.append(f"- {report.symbol}: elevated volatility {report.annualized_volatility:.1%}")

    lines.extend(
        [
            "",
            "## Recommendations (rule-based)",
            "- Review regime routing when ADX < 20 (range market mean-reversion bias).",
            "- Reduce turnover if execution costs exceed 10% of gross return.",
            "- Add sector/symbol concentration limits if single-name exposure is high.",
            "",
            "## Next Steps",
            "- [ ] Backtest ADX filter changes",
            "- [ ] Validate sentiment features vs price momentum",
            "- [ ] Human review of LLM narrative sections",
        ]
    )
    return "\n".join(lines)

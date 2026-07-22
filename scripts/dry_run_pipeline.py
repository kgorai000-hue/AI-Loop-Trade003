from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.pipeline import TradingPipeline
from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.survival.reporting import print_survival_report


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strategy lifecycle pipeline in dry-run mode (Lesson 02-03)"
    )
    parser.add_argument("--symbol", action="append", help="Limit pipeline to specific symbols")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)

    if not config.trading.dry_run:
        print("ERROR: trading.dry_run must be true for this script")
        return 1

    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    pipeline = TradingPipeline(config, connector, store)

    try:
        connector.connect()

        print("\n=== Trading Pipeline (Dry Run) ===")
        print(f"  profile         : {config.trading.profile}")
        analysis_tf = config.stats.analysis_timeframe
        signal_tf = config.stats.signal_timeframe
        print(f"  stats TF        : {analysis_tf} | signal TF: {signal_tf}")
        print(f"  dry_run         : {config.trading.dry_run}")

        result = pipeline.run(symbols=args.symbol)

        if result.market_regime:
            mr = result.market_regime
            print(f"\n=== Market Regime ({mr.symbol}) ===")
            print(f"  regime          : {mr.regime.value} ({mr.regime_label})")
            print(f"  ann vol         : {mr.annualized_volatility:.2%}")
            print(f"  ADX             : {mr.adx:.1f}")
            print(f"  correlation     : {mr.asset_correlation:.2f}")
            print(f"  strategy        : {mr.selected_strategy.value}")
            print(f"  position scale  : {mr.position_scale:.0%}")
            if mr.strategy_weights:
                weights = ", ".join(f"{k}={v:.0%}" for k, v in mr.strategy_weights.items())
                print(f"  weights         : {weights}")
            print(f"  recommended     : {mr.recommended_mode.value}")
            print(f"  reason          : {mr.reason}")

        print(f"\n=== Research Summary ({len(result.research_reports)}) ===")
        for report in result.research_reports[:5]:
            rsi = f"{report.rsi:.1f}" if report.rsi is not None else "n/a"
            print(
                f"  {report.symbol:12} vol={report.annualized_volatility:6.2%} "
                f"RSI={rsi:>5} regime={report.regime.value} "
                f"div={report.macd_divergence or '-'}"
            )
        if len(result.research_reports) > 5:
            print(f"  ... and {len(result.research_reports) - 5} more")

        print(f"\n=== Raw Signals ({len(result.raw_signals)}) ===")
        for signal in result.raw_signals:
            pred = signal.predicted_return if signal.predicted_return is not None else signal.strength * 0.01
            conf = signal.confidence if signal.confidence is not None else signal.strength
            print(
                f"  {signal.symbol:12} {signal.side.value:4} {signal.strategy.value:18} "
                f"pred={pred:+.3%} conf={conf:.2f} | {signal.reason}"
            )

        print(f"\n=== Decisions ({len(result.decision_reports)}) ===")
        for report in result.decision_reports:
            print(
                f"  {report.symbol:12} {report.side.value:4} "
                f"lots={report.requested_lots:.2f} final={report.final_position_pct:.2f}% | "
                f"kelly={report.half_kelly_cap_pct:.1f}% van_tharp={report.van_tharp_cap_pct:.1f}%"
            )

        print(f"\n=== Sized Signals ({len(result.signals)}) ===")
        for signal in result.signals:
            lots = signal.requested_lots if signal.requested_lots is not None else 0.0
            print(
                f"  {signal.symbol:12} {signal.side.value:4} {signal.strategy.value:18} "
                f"requested_lots={lots:.2f} | {signal.reason}"
            )

        print(f"\n=== Risk Decisions ({len(result.risk_decisions)}) ===")
        for risk in result.risk_decisions:
            print(f"  {risk.decision.value:7} lots={risk.approved_lots:.2f} | {risk.reason}")

        print(f"\n=== Execution Plans ({len(result.execution_plans)}) ===")
        for plan in result.execution_plans:
            fill_price = plan.average_fill_price if plan.average_fill_price else 0.0
            print(
                f"  {plan.symbol:12} {plan.side.value:4} {plan.lots:.2f} lots "
                f"{plan.order_type:5} {plan.algo:10} status={plan.status:8} "
                f"fill={plan.fill_ratio:.0%} slip={plan.slippage_pct:.3f}% "
                f"latency={plan.latency_ms:.0f}ms @ {fill_price:.5f}"
            )

        print(f"\n=== Position Actions ({len(result.position_actions)}) ===")
        if result.position_actions:
            for action in result.position_actions:
                print(f"  {action.symbol:12} {action.action:18} | {action.reason}")
        else:
            print("  (no open positions or no exit triggers)")

        if result.hedge_report:
            hr = result.hedge_report
            print(f"\n=== Hedging / Beta (Lesson 08) ===")
            print(f"  benchmark         : {hr.benchmark}")
            print(f"  portfolio beta    : {hr.portfolio_beta:.2f}")
            print(f"  dollar-neutral net beta : {hr.dollar_neutral_net_beta:,.0f}")
            print(f"  beta-neutral hedge      : {hr.beta_neutral_hedge_notional:,.0f} JPY")
            if hr.recommendation:
                rec = hr.recommendation
                print(f"  recommendation    : {rec.reason}")
                print(f"  net beta before/after: {rec.net_beta_before:,.0f} -> {rec.net_beta_after:,.0f}")
                print(f"  hedge cost (ann)  : {rec.hedge_cost_annual_pct:.2%} of capital")
            else:
                print("  recommendation    : (within beta tolerance, no hedge needed)")
            for note in hr.notes[:3]:
                print(f"  note              : {note}")

        if result.multi_agent_report:
            mr = result.multi_agent_report
            print(f"\n=== Multi-Agent (Lesson 11) ===")
            print(f"  evolution stage   : {mr.evolution_stage}")
            print(f"  arbitration       : {mr.arbitration_mode}")
            print(f"  parallel analysis : {mr.parallel_analysis}")
            print(f"  elapsed           : {mr.parallel_elapsed_ms:.0f} ms (serial est. {mr.serial_estimate_ms:.0f} ms)")
            print(f"  bus events        : {mr.bus_events}")
            if mr.arbitration_results:
                print(f"  arbitration notes : {len(mr.arbitration_results)}")
                for note in mr.arbitration_results[:3]:
                    print(f"    {note.symbol}: {note.reason}")
            unhealthy = [h for h in mr.agent_health if not h.healthy or h.circuit_open]
            if unhealthy:
                print("  agent health issues:")
                for h in unhealthy:
                    print(f"    {h.agent}: circuit={h.circuit_open} err={h.last_error}")
            else:
                print("  agent health      : all agents healthy")

        if result.resilience_report:
            rr = result.resilience_report
            print(f"\n=== Resilience (Lesson 13) ===")
            print(f"  degradation level : {rr.level_name} ({rr.degradation_level})")
            print(f"  position scale    : {rr.position_scale_multiplier:.0%}")
            if rr.warnings:
                for warning in rr.warnings[:3]:
                    print(f"  warning           : {warning}")

        if result.llm_research_report:
            lr = result.llm_research_report
            print(f"\n=== LLM Research (Lesson 14) ===")
            print(f"  provider          : {lr.provider}")
            print(f"  news filtered     : {lr.filtered_news_count}")
            print(f"  analyzed          : {lr.analyzed_count} (skipped {lr.skipped_count})")
            print(f"  audit records     : {len(lr.audit_ids)}")
            for feat in lr.sentiment_features[:5]:
                print(
                    f"  {feat.symbol}: sentiment={feat.sentiment_score:+.2f} "
                    f"events={feat.event_count} conf={feat.confidence:.2f}"
                )
            if lr.strategy_report:
                print(f"  strategy report   : {len(lr.strategy_report.splitlines())} lines")

        if result.risk_control_report:
            rc = result.risk_control_report
            print(f"\n=== Risk Control (Lesson 15) ===")
            print(f"  equity            : {rc.current_equity:,.0f} (peak {rc.peak_equity:,.0f})")
            print(f"  drawdown          : {rc.drawdown_pct:.1f}% ({rc.drawdown_level})")
            print(f"  action            : {rc.drawdown_action}")
            print(f"  position scale    : {rc.position_scale:.0%}")
            print(f"  total exposure    : {rc.total_exposure_pct:.1f}%")
            print(f"  circuit breaker   : {rc.circuit_breaker_active}")
            print(f"  new positions     : {rc.new_positions_allowed}")
            for warning in rc.warnings[:3]:
                print(f"  warning           : {warning}")

        if result.portfolio_report:
            pr = result.portfolio_report
            print(f"\n=== Portfolio Construction (Lesson 16) ===")
            print(f"  weight method     : {pr.weight_method}")
            print(f"  portfolio vol     : {pr.portfolio_volatility:.1%}")
            print(f"  risk leverage     : {pr.risk_leverage:.2f}x")
            print(f"  avg correlation   : {pr.avg_correlation:.2f}")
            if pr.shrinkage is not None:
                print(f"  shrinkage delta   : {pr.shrinkage:.3f}")
            for alloc in pr.allocations[:5]:
                print(
                    f"  {alloc.symbol}: weight={alloc.adjusted_weight:.1%} "
                    f"vol={alloc.annualized_volatility:.1%} strength={alloc.signal_strength:.2f}"
                )
            for exposure in pr.factor_exposures[:4]:
                flag = " BREACH" if exposure.breached else ""
                print(f"  factor {exposure.factor}: {exposure.exposure:.2f}{flag}")

        if result.evolution_report:
            er = result.evolution_report
            print(f"\n=== Online Learning / Evolution (Lesson 17) ===")
            print(f"  mean IC           : {er.mean_ic:.3f}")
            print(f"  projected IC 12m  : {er.projected_ic_12m:.3f}")
            print(f"  effective lookback: {er.effective_lookback_days} days")
            print(f"  dynamic threshold : {er.dynamic_threshold:.3f}")
            print(f"  drift detected    : {er.drift_detected}")
            if er.update_decision:
                ud = er.update_decision
                print(f"  update action     : {ud.action.value} ({ud.confidence:.0%})")
                print(f"  reason            : {ud.reason}")
            for state in er.strategy_states[:4]:
                print(
                    f"  {state.strategy}: stage={state.stage.value} "
                    f"sharpe={state.sharpe_proxy:.2f} weight={state.capital_weight:.0%}"
                )
            for warning in er.warnings[:3]:
                print(f"  warning           : {warning}")

        if result.cost_report:
            cr = result.cost_report
            print(f"\n=== Trading Costs / Tradability (Lesson 18) ===")
            print(f"  assessments       : {len(cr.assessments)}")
            print(f"  blocked           : {cr.blocked_count}")
            print(f"  total est. cost   : {cr.total_estimated_cost_jpy:,.2f} JPY")
            for assessment in cr.assessments[:5]:
                print(
                    f"  {assessment.symbol:12} gross={assessment.gross_alpha_pct:+.3f}% "
                    f"cost={assessment.costs.total_pct:.3f}% "
                    f"net={assessment.net_alpha_pct:+.3f}% "
                    f"tradable={assessment.tradable} fill={assessment.fill_probability:.0%}"
                )
            for warning in cr.warnings[:3]:
                print(f"  warning           : {warning}")

        if result.execution_report:
            er = result.execution_report
            print(f"\n=== Execution System / Telemetry (Lesson 19) ===")
            print(f"  logged records    : {len(er.records)}")
            print(f"  avg slippage      : {er.avg_slippage_pct:.4f}%")
            print(f"  avg fill ratio    : {er.avg_fill_ratio:.0%}")
            print(f"  avg latency       : {er.avg_latency_ms:.0f} ms")
            print(f"  partial fills     : {er.partial_fill_count}")
            for warning in er.warnings[:3]:
                print(f"  warning           : {warning}")

        if result.trade_log_report:
            tl = result.trade_log_report
            print(f"\n=== Appendix A Trade Log ===")
            print(f"  orders (sample)   : {tl.order_count}")
            print(f"  total fills       : {tl.fill_count}")
            print(f"  avg slippage      : {tl.avg_slippage_pct:.4f}%")
            print(f"  avg fill ratio    : {tl.avg_fill_ratio:.0%}")
            print(f"  avg latency       : {tl.avg_latency_ms:.0f} ms")
            print(f"  total commission  : {tl.total_commission:.2f} JPY")

        if result.ops_report:
            op = result.ops_report
            print(f"\n=== Production Operations (Lesson 20) ===")
            print(f"  session phase     : {op.session_phase}")
            print(f"  trading allowed   : {op.trading_allowed}")
            print(f"  healthy           : {op.healthy}")
            print(f"  critical alerts   : {op.critical_count}")
            unhealthy = [m for m in op.metrics if not m.healthy]
            print(f"  metric breaches   : {len(unhealthy)}")
            for item in op.checklist[:5]:
                flag = "OK" if item.passed else "FAIL"
                print(f"  [{flag}] {item.phase:12} {item.name:20} | {item.detail}")
            for alert in op.alerts[:3]:
                if not alert.suppressed:
                    print(f"  alert             : [{alert.severity.value}] {alert.title}")
            for warning in op.warnings[:3]:
                print(f"  warning           : {warning}")

        if result.survival_report:
            print_survival_report(result.survival_report, compact=True)

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

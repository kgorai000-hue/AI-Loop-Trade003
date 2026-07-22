from __future__ import annotations

from collections import defaultdict

from src.core.types import (
    ArbitrationMode,
    ArbitrationResult,
    AgentVote,
    RegimeAssessment,
    SignalMode,
    SignalSide,
    StrategyKind,
    TradeSignal,
)


def resolve_signal_conflicts(
    signals: list[TradeSignal],
    mode: ArbitrationMode,
) -> tuple[list[TradeSignal], list[ArbitrationResult]]:
    """Resolve same-symbol conflicts before sizing (Lesson 11.3 hierarchy)."""
    if mode == ArbitrationMode.VETO:
        return signals, []

    grouped: dict[str, list[TradeSignal]] = defaultdict(list)
    for signal in signals:
        grouped[signal.symbol].append(signal)

    resolved: list[TradeSignal] = []
    notes: list[ArbitrationResult] = []

    for symbol, group in grouped.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        sides = {signal.side for signal in group}
        if len(sides) == 1:
            best = max(group, key=lambda s: s.strength)
            resolved.append(best)
            notes.append(
                ArbitrationResult(
                    symbol=symbol,
                    approved=True,
                    net_score=1,
                    votes=[AgentVote("meta", 1, f"hierarchy: kept strongest of {len(group)}")],
                    selected_signal=best,
                    reason="same-side conflict resolved by strength",
                )
            )
            continue

        if mode == ArbitrationMode.HIERARCHY:
            best = max(group, key=lambda s: s.strength)
            resolved.append(best)
            notes.append(
                ArbitrationResult(
                    symbol=symbol,
                    approved=True,
                    net_score=1,
                    votes=[AgentVote("meta", 1, "hierarchy override")],
                    selected_signal=best,
                    reason="opposing signals; meta selected highest strength",
                )
            )
            continue

        # voting: count buy vs sell weighted by strength
        buy_score = sum(s.strength for s in group if s.side == SignalSide.BUY)
        sell_score = sum(s.strength for s in group if s.side == SignalSide.SELL)
        net = 1 if buy_score > sell_score else (-1 if sell_score > buy_score else 0)
        votes = [
            AgentVote("signal", net, f"buy={buy_score:.2f} vs sell={sell_score:.2f}"),
        ]
        if net == 0:
            notes.append(
                ArbitrationResult(
                    symbol=symbol,
                    approved=False,
                    net_score=0,
                    votes=votes,
                    reason="vote tie; no trade",
                )
            )
            continue

        target_side = SignalSide.BUY if net > 0 else SignalSide.SELL
        candidates = [s for s in group if s.side == target_side]
        best = max(candidates, key=lambda s: s.strength)
        resolved.append(best)
        notes.append(
            ArbitrationResult(
                symbol=symbol,
                approved=True,
                net_score=net,
                votes=votes,
                selected_signal=best,
                reason="voting resolved opposing signals",
            )
        )

    return resolved, notes


def vote_on_proposal(
    signal: TradeSignal,
    regime: RegimeAssessment | None,
    exposure_pct: float,
    max_exposure_pct: float,
) -> ArbitrationResult:
    """Lesson 11.3 voting example: Signal + Risk + Regime."""
    votes: list[AgentVote] = [
        AgentVote("signal", 1, f"{signal.side.value} proposal from {signal.strategy.value}"),
    ]

    if regime is not None:
        if regime.selected_strategy == StrategyKind.CRISIS_HALT:
            votes.append(AgentVote("regime", -1, "crisis halt"))
        elif signal.mode == SignalMode.MOMENTUM and regime.recommended_mode == SignalMode.MOMENTUM:
            votes.append(AgentVote("regime", 1, "trend regime aligned"))
        elif signal.mode == SignalMode.MEAN_REVERSION and regime.recommended_mode == SignalMode.MEAN_REVERSION:
            votes.append(AgentVote("regime", 1, "range regime aligned"))
        elif regime.recommended_mode == SignalMode.NONE:
            votes.append(AgentVote("regime", 0, "neutral regime"))
        else:
            votes.append(AgentVote("regime", -1, "regime/signal mode mismatch"))

    if exposure_pct >= max_exposure_pct:
        votes.append(AgentVote("risk", -1, f"exposure {exposure_pct:.1f}% >= max"))
    else:
        votes.append(AgentVote("risk", 1, "within exposure limits"))

    net = sum(vote.vote for vote in votes)
    risk_veto = any(vote.agent == "risk" and vote.vote < 0 for vote in votes)
    approved = net > 0 and not risk_veto
    reason = "approved by vote" if approved else (
        "risk veto" if risk_veto else f"rejected by vote (net={net})"
    )
    return ArbitrationResult(
        symbol=signal.symbol,
        approved=approved,
        net_score=net,
        votes=votes,
        selected_signal=signal if approved else None,
        reason=reason,
    )

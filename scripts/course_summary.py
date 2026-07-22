from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_INSIGHTS = [
    ("1", "Prediction != Profit", "Good IC still loses after costs, execution, risk"),
    ("2", "No single model for all regimes", "Multi-agent specialization is the answer"),
    ("3", "Risk must have veto power", "Independent RiskAgent, not embedded in signals"),
    ("4", "Backtest != Live", "Backtest is a filter, not an oracle (quality gates)"),
    ("5", "Systems decay", "Online learning and evolution required (Lesson 17)"),
]

MISCONCEPTIONS = [
    ("50% backtest return = live profit", "Subtract costs, pass OOS, paper trade first"),
    ("60% accuracy is impressive", "Check IC, turnover, and net alpha after costs"),
    ("Deep learning beats simple models", "Start simple; complexity overfits easily"),
    ("Diversification = many symbols", "High correlation != true diversification"),
    ("Stop-loss hurts returns", "Uncontrolled drawdown destroys capital"),
    ("Strategy is finished", "Markets change; monitor drift and evolve"),
    ("LLM can trade directly", "LLM is research assistant only (see guard.py)"),
    ("Code runs = system is reliable", "Ops monitoring and alerts are mandatory"),
]

LLM_BOUNDARY = [
    ("CAN", "Parse filings, summarize news, flag anomalies, generate hypotheses"),
    ("CANNOT", "Submit orders, size positions, override risk, replace quant models"),
]

KNOWLEDGE_QUIZ = [
    "Why multi-agent instead of one model?",
    "What is Risk Agent's core design principle?",
    "Name 3 backtest pitfalls and fixes.",
    "What role should LLM play in a quant system?",
    "What causes strategy decay and how to address it?",
]

KEY_FORMULAS = [
    "Live Return = Strategy Return - Cost - Slippage - Impact",
    "Kelly: f = (pb - q) / b",
    "IC = corr(Prediction, Actual)",
    "Sharpe = (Return - Rf) / Volatility",
    "Net Regime Value = Return Improvement - Switching Cost",
]


def print_lesson_content() -> None:
    print("\n=== Lesson 22: Summary and Advanced Directions ===")
    print("  Course complete. Core insights, misconceptions, and next steps.\n")

    print("=== 5 Core Insights (22.1) ===")
    for num, title, detail in CORE_INSIGHTS:
        print(f"  {num}. {title}")
        print(f"     {detail}")

    print("\n=== Key Formulas ===")
    for formula in KEY_FORMULAS:
        print(f"  - {formula}")

    print("\n=== Misconception Checklist (22.2) ===")
    for wrong, right in MISCONCEPTIONS:
        print(f"  X {wrong}")
        print(f"    -> {right}")

    print("\n=== LLM Boundary (22.2) ===")
    for label, items in LLM_BOUNDARY:
        print(f"  {label}: {items}")
    print("  Enforced in: src/llm_research/guard.py (feature-only pipeline integration)")

    print("\n=== Knowledge Quiz (self-check, 22 Course Complete) ===")
    for idx, question in enumerate(KNOWLEDGE_QUIZ, 1):
        print(f"  {idx}. {question}")

    print("\n=== Practice Checklist ===")
    checks = [
        "Lesson 21 project runs: python main.py run",
        "Backtest validation: scripts/run_backtest_validation.py",
        "Appendix A trade log: python scripts/trade_log_report.py",
        "Appendix B survival: python scripts/death_modes_report.py",
        "Ops checklist: python main.py status",
        "Extension stubs reserved: src/extensions/ (HFT, alt data, distributed)",
    ]
    for item in checks:
        print(f"  [ ] {item}")

    print("\n=== Advanced Paths (reference only, 22.4) ===")
    print("  Path 1 Technical: HFT microstructure, low-latency (src/extensions/hft.py)")
    print("  Path 2 Strategy: factors, alt data (src/extensions/alt_data.py)")
    print("  Path 3 Career: quant researcher / developer / independent trader")
    print("  Extract to services when: see src/extensions/distributed.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Course summary and completion guide (Lesson 22)")
    parser.add_argument("--paper-only", action="store_true", help="Print summary (default)")
    args = parser.parse_args()

    print_lesson_content()

    if not args.paper_only:
        from src.extensions.hft import LatencyBudget

        print("\n=== Production Benchmark Gap (22.4) ===")
        for key, line in LatencyBudget.gap_report().items():
            print(f"  {key}: {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

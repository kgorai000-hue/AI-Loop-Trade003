from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.data.store import OHLCVStore
from src.ml.models import recommend_model_type
from src.ml.trainer import MLTrainer


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_report(report, ic_threshold: float) -> None:
    print(f"\n--- {report.symbol} ({report.timeframe}) ---")
    print(f"  model             : {report.model_type}")
    print(f"  samples           : {report.n_samples}")
    print(f"  features          : {report.n_features} {report.selected_features}")
    print(f"  mean IC           : {report.mean_ic:.4f} (threshold {ic_threshold})")
    print(f"  mean IR           : {report.mean_ir:.2f}")
    print(f"  mean accuracy     : {report.mean_accuracy:.2%}")
    print(f"  viable signal     : {report.viable}")
    if report.ic_decay_warning:
        print("  WARNING           : IC decay detected")
    if report.top_feature_importance:
        top = ", ".join(f"{n}={v:.0%}" for n, v in report.top_feature_importance[:3])
        print(f"  top importance    : {top}")
    print("  fold IC           :", ", ".join(f"{f.ic:+.3f}" for f in report.fold_results))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate ML model with walk-forward IC (Lesson 09)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override signal timeframe")
    parser.add_argument("--symbols", action="append", help="Train multiple symbols")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    trainer = MLTrainer(config)
    timeframe = args.timeframe or config.stats.signal_timeframe

    symbols = args.symbols or ([args.symbol] if args.symbol else config.symbols[:3])
    print(f"\n=== ML Training Report (Lesson 09) ===")
    print(f"  label horizon     : {config.ml.label_horizon_bars} bars")
    print(f"  label threshold   : {config.ml.label_threshold:.3%}")
    print(f"  recommended model : {recommend_model_type(config.history_bars_for(timeframe), config.ml)}")

    viable_count = 0
    for symbol in symbols:
        bars = store.get_recent_bars(symbol, timeframe, config.history_bars_for(timeframe))
        if len(bars) < config.stats.min_bars:
            print(f"\n--- {symbol} --- insufficient bars")
            continue
        try:
            report = trainer.train_and_evaluate(bars, symbol, timeframe)
            print_report(report, config.ml.ic_threshold)
            if report.viable:
                viable_count += 1
        except ValueError as exc:
            print(f"\n--- {symbol} --- skipped: {exc}")

    print(f"\n=== Summary ===")
    print(f"  symbols trained   : {len(symbols)}")
    print(f"  viable (IC/IR)    : {viable_count}")
    print("  Note: IC > 0.03 is strong for quant; accuracy alone is misleading.")
    print("  Enable pipeline ML: set ml.enabled: true in settings.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

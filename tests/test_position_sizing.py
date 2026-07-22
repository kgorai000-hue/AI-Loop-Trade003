from __future__ import annotations

import unittest

from src.execution.position_sizing import half_kelly, risk_parity_weights, van_tharp_cap_pct


class PositionSizingTests(unittest.TestCase):
    def test_half_kelly_lesson_example(self) -> None:
        # win 55%, reward/risk 1.5 -> full kelly 25%, half 12.5%
        self.assertAlmostEqual(half_kelly(0.55, 1.5), 0.125, places=3)

    def test_van_tharp_lesson_example(self) -> None:
        # $100k, 1% risk, $10 stop, $200 price -> 20% cap
        cap = van_tharp_cap_pct(100_000, 0.01, 10.0, 200.0)
        self.assertAlmostEqual(cap, 0.20, places=3)

    def test_risk_parity_scenario(self) -> None:
        vols = {"AAPL": 0.25, "TSLA": 0.50, "MSFT": 0.20}
        weights = risk_parity_weights(vols)
        self.assertAlmostEqual(weights["AAPL"], 4 / 11, places=3)
        self.assertAlmostEqual(weights["TSLA"], 2 / 11, places=3)
        self.assertAlmostEqual(weights["MSFT"], 5 / 11, places=3)


if __name__ == "__main__":
    unittest.main()

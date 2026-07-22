from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


@dataclass
class SymbolStatus:
    requested: str
    resolved: str | None
    available: bool
    bid: float | None = None
    ask: float | None = None
    spread_points: int | None = None
    error: str | None = None


class MT5Connector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return

        mt5_cfg = self.config.mt5
        init_kwargs: dict[str, Any] = {"timeout": mt5_cfg.timeout_ms}
        if mt5_cfg.path:
            init_kwargs["path"] = mt5_cfg.path

        if not mt5.initialize(**init_kwargs):
            raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

        if mt5_cfg.login and mt5_cfg.password and mt5_cfg.server:
            authorized = mt5.login(
                login=int(mt5_cfg.login),
                password=mt5_cfg.password,
                server=mt5_cfg.server,
            )
            if not authorized:
                mt5.shutdown()
                raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")

        account = mt5.account_info()
        if account is None:
            mt5.shutdown()
            raise ConnectionError(f"MT5 account_info unavailable: {mt5.last_error()}")

        self._assert_demo_account(account)

        self._connected = True
        logger.info(
            "Connected to MT5: login=%s server=%s balance=%.2f %s trade_mode=%s",
            account.login,
            account.server,
            account.balance,
            account.currency,
            getattr(account, "trade_mode", "?"),
        )

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed")

    def __enter__(self) -> MT5Connector:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def get_account_info(self) -> dict[str, Any]:
        self._ensure_connected()
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")

        return account._asdict()

    def resolve_symbol(self, symbol: str) -> str | None:
        self._ensure_connected()

        if mt5.symbol_info(symbol) is not None:
            return symbol

        matches = mt5.symbols_get(group=f"*{symbol}*") or []
        for candidate in matches:
            if candidate.name == symbol or candidate.name.startswith(symbol):
                return candidate.name

        return None

    def ensure_symbol(self, symbol: str) -> str:
        resolved = self.resolve_symbol(symbol)
        if resolved is None:
            raise ValueError(f"Symbol not found: {symbol}")

        if not mt5.symbol_select(resolved, True):
            raise RuntimeError(f"symbol_select failed for {resolved}: {mt5.last_error()}")

        return resolved

    def check_symbols(self, symbols: list[str]) -> list[SymbolStatus]:
        self._ensure_connected()
        results: list[SymbolStatus] = []

        for symbol in symbols:
            try:
                resolved = self.resolve_symbol(symbol)
                if resolved is None:
                    results.append(
                        SymbolStatus(
                            requested=symbol,
                            resolved=None,
                            available=False,
                            error="symbol not found",
                        )
                    )
                    continue

                self.ensure_symbol(resolved)
                tick = mt5.symbol_info_tick(resolved)
                info = mt5.symbol_info(resolved)

                if tick is None or info is None:
                    results.append(
                        SymbolStatus(
                            requested=symbol,
                            resolved=resolved,
                            available=False,
                            error="tick/info unavailable",
                        )
                    )
                    continue

                results.append(
                    SymbolStatus(
                        requested=symbol,
                        resolved=resolved,
                        available=True,
                        bid=tick.bid,
                        ask=tick.ask,
                        spread_points=info.spread,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report per-symbol failures
                results.append(
                    SymbolStatus(
                        requested=symbol,
                        resolved=None,
                        available=False,
                        error=str(exc),
                    )
                )

        return results

    def get_rates(self, symbol: str, timeframe: str, count: int) -> Any:
        self._ensure_connected()
        resolved = self.ensure_symbol(symbol)

        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")

        return rates

    def get_rates_from(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        count: int,
    ) -> Any:
        self._ensure_connected()
        resolved = self.ensure_symbol(symbol)

        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_from(resolved, tf, date_from, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from failed: {mt5.last_error()}")

        return rates

    def get_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> Any:
        self._ensure_connected()
        resolved = self.ensure_symbol(symbol)

        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_range(resolved, tf, date_from, date_to)
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed: {mt5.last_error()}")

        return rates

    def health_check(self) -> dict[str, Any]:
        self._ensure_connected()
        terminal = mt5.terminal_info()
        account = mt5.account_info()

        if terminal is None or account is None:
            raise RuntimeError(f"health_check failed: {mt5.last_error()}")

        return {
            "terminal_connected": terminal.connected,
            "terminal_trade_allowed": terminal.trade_allowed,
            "account_login": account.login,
            "account_server": account.server,
            "account_balance": account.balance,
            "account_equity": account.equity,
            "account_trade_allowed": account.trade_allowed,
            "account_trade_mode": getattr(account, "trade_mode", None),
            "is_demo": self.is_demo_account(account),
        }

    def order_send(self, request: dict[str, Any]) -> Any:
        """Send an MT5 trade request. Caller must interpret None / retcodes."""
        self._ensure_connected()
        return mt5.order_send(request)

    def positions_get(self, symbol: str | None = None) -> tuple[Any, ...] | None:
        self._ensure_connected()
        if symbol:
            return mt5.positions_get(symbol=symbol)
        return mt5.positions_get()

    def symbol_info(self, symbol: str) -> Any:
        self._ensure_connected()
        resolved = self.ensure_symbol(symbol)
        return mt5.symbol_info(resolved)

    def symbol_info_tick(self, symbol: str) -> Any:
        self._ensure_connected()
        resolved = self.ensure_symbol(symbol)
        return mt5.symbol_info_tick(resolved)

    def last_error(self) -> Any:
        return mt5.last_error()

    @staticmethod
    def is_demo_account(account: Any) -> bool:
        trade_mode = getattr(account, "trade_mode", None)
        # ACCOUNT_TRADE_MODE_DEMO == 0 in MetaTrader5
        if trade_mode is not None and int(trade_mode) == 0:
            return True
        server = str(getattr(account, "server", "") or "").lower()
        if "demo" in server:
            return True
        return False

    def _assert_demo_account(self, account: Any) -> None:
        require_demo = bool(getattr(self.config.mt5, "require_demo", True))
        account_type = str(self.config.account_type or "").lower()
        if not require_demo and account_type != "demo":
            return
        if self.is_demo_account(account):
            return
        mt5.shutdown()
        raise ConnectionError(
            "Refusing to trade: MT5 account is not DEMO. "
            "Set broker.account_type=demo and mt5.require_demo=true for Phase 1."
        )

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MT5 is not connected. Call connect() first.")


def create_connector(config: AppConfig) -> MT5Connector:
    return MT5Connector(config)

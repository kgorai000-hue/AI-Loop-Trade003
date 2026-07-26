"""Unit tests for Trade002 intelligence param helpers and state store."""

from __future__ import annotations

from pathlib import Path

from src.intelligence.anthropic_client import AnthropicClient
from src.intelligence.maker import StrategyMaker
from src.intelligence.params import (
    canonicalize_candidate,
    candidate_key,
    llm_parameter_specs,
    params_from_config,
)
from src.intelligence.persistence import StateStore
from src.core.config import load_config


def test_anthropic_key_shape_rejects_placeholder():
    assert AnthropicClient.looks_like_api_key(None) is False
    assert AnthropicClient.looks_like_api_key("") is False
    assert AnthropicClient.looks_like_api_key("sk-ant-...") is False
    assert AnthropicClient.looks_like_api_key("not-a-key") is False
    assert AnthropicClient.looks_like_api_key("sk-ant-api03-abcdefghijklmnop") is True


def test_llm_specs_exclude_trading_profile():
    names = {s.name for s in llm_parameter_specs()}
    assert "trading_profile" not in names
    assert "signal_score_threshold" in names


def test_canonicalize_signal_score():
    cleaned = canonicalize_candidate({"indicators.signal_score_threshold": 0.15})
    assert cleaned == {"indicators.signal_score_threshold": 0.15}

    bad = canonicalize_candidate({"indicators.signal_score_threshold": 0.99})
    assert bad is None

    unknown = canonicalize_candidate({"strategies.trend_ma_short": 10})
    assert unknown is None


def test_canonicalize_rsi_pair():
    cleaned = canonicalize_candidate(
        {
            "strategies.mr_rsi_oversold": 25.0,
            "strategies.mr_rsi_overbought": 75.0,
        }
    )
    assert cleaned is not None
    assert cleaned["strategies.mr_rsi_oversold"] == 25.0


def test_maker_parse_candidates():
    maker = StrategyMaker(client=None)  # type: ignore[arg-type]
    data = {
        "candidates": [
            {
                "overrides": {"indicators.signal_score_threshold": 0.20},
                "rationale": "test",
            },
            {
                "overrides": {"indicators.signal_score_threshold": 0.99},
                "rationale": "invalid",
            },
        ]
    }
    parsed = maker._parse_candidates(data)
    assert len(parsed) == 1
    assert parsed[0].params.as_dict()["indicators.signal_score_threshold"] == 0.20


def test_state_store_roundtrip(tmp_path: Path):
    store = StateStore(tmp_path, "#US30")
    store.update_state(
        params={"indicators.signal_score_threshold": 0.25},
        accepted=True,
        strategy="feature_score",
    )
    params = store.get_params()
    assert params.as_dict()["indicators.signal_score_threshold"] == 0.25
    store.append_lesson("Checker reject demo")
    assert "Checker reject demo" in store.skills_text()


def test_params_from_config_loads():
    config = load_config()
    params = params_from_config(config)
    assert "indicators.signal_score_threshold" in params.as_dict()
    assert candidate_key(params.as_dict())


def test_load_config_intelligence_section():
    config = load_config()
    assert config.intelligence.enabled is True
    assert config.intelligence.state_dir == "state"
    assert config.trading.primary_timeframe == "M30"

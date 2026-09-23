"""HTTP-клиент шлюза: ретраи, коды ошибок, учёт стоимости. requests.post мокается —
реальная сеть в юнит-тестах не участвует. time.sleep мокается, чтобы ретраи (экспоненциальный
бэкофф до 5 попыток) не делали тесты медленными."""
from __future__ import annotations

import pytest

import swarm.gateway as gw
from swarm.gateway import CallResult, FakeGateway, Gateway, GatewayError, ModelUnavailable, Usage


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gw.time, "sleep", lambda *a, **kw: None)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("EXPLABS_API_KEY", "xpl_test_key")


class FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text or str(json_body)

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def _ok_response(text="hello", input_tokens=10, output_tokens=5, cost=0.001, stop_reason="end_turn"):
    return FakeResponse(200, {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost},
        "stop_reason": stop_reason,
    })


# ---------- CallResult / Usage ----------

def test_call_result_shadow_cost_known_model():
    r = CallResult(text="x", input_tokens=1_000_000, output_tokens=1_000_000, cost=0.0)
    assert r.shadow_cost("claude-sonnet-5") == pytest.approx(3.0 + 15.0)


def test_call_result_shadow_cost_unknown_model_uses_default_bracket():
    r = CallResult(text="x", input_tokens=1_000_000, output_tokens=1_000_000, cost=0.0)
    assert r.shadow_cost("some-unlisted-model") == pytest.approx(1.0 + 3.0)


def test_usage_add_accumulates_totals_and_per_model():
    u = Usage()
    u.add("claude-sonnet-5", CallResult(text="a", input_tokens=100, output_tokens=50, cost=0.01))
    u.add("claude-sonnet-5", CallResult(text="b", input_tokens=200, output_tokens=100, cost=0.02))
    assert u.calls == 2
    assert u.input_tokens == 300
    assert u.output_tokens == 150
    assert u.actual_cost == pytest.approx(0.03)
    assert u.by_model["claude-sonnet-5"]["calls"] == 2


def test_usage_report_contains_totals_and_model_rows():
    u = Usage()
    u.add("gpt-6-sol", CallResult(text="a", input_tokens=100, output_tokens=50, cost=0.0))
    report = u.report()
    assert "Вызовов всего" in report
    assert "gpt-6-sol" in report


# ---------- Gateway.__init__ ----------

def test_gateway_without_key_raises(monkeypatch):
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    with pytest.raises(GatewayError, match="EXPLABS_API_KEY"):
        Gateway()


# ---------- Gateway.call: pre-HTTP guards ----------

def test_gateway_call_unavailable_model_raises_without_http(key, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    g = Gateway()
    with pytest.raises(ModelUnavailable, match="claude-fable-5.1"):
        g.call("claude-fable-5.1", "hi")
    assert called["n"] == 0


def test_gateway_call_dead_model_raises_without_http(key, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    g = Gateway()
    g.dead_models.add("gpt-6-sol")
    with pytest.raises(ModelUnavailable, match="gpt-6-sol"):
        g.call("gpt-6-sol", "hi")
    assert called["n"] == 0


def test_gateway_call_gpt_family_bumps_low_max_tokens(key, monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["max_tokens"] = json["max_tokens"]
        return _ok_response()

    monkeypatch.setattr(gw.requests, "post", fake_post)
    g = Gateway()
    g.call("gpt-6-sol", "hi", max_tokens=4)
    assert captured["max_tokens"] == 16  # раздел 3: GPT-семейство требует >= 16


# ---------- Gateway.call: HTTP status handling ----------

def test_gateway_call_success_returns_result_and_updates_usage(key, monkeypatch):
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: _ok_response(text="реальный ответ"))
    g = Gateway()
    result = g.call("gpt-6-sol", "hi")
    assert result.text == "реальный ответ"
    assert result.input_tokens == 10
    assert g.usage.calls == 1


def test_gateway_call_429_model_requires_purchase_raises_unavailable(key, monkeypatch):
    resp = FakeResponse(429, {"error": {"code": "model_requires_purchase"}})
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: resp)
    g = Gateway()
    with pytest.raises(ModelUnavailable, match="тариф"):
        g.call("gpt-6-sol", "hi")


def test_gateway_call_429_generic_retries_then_raises(key, monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **kw):
        calls["n"] += 1
        return FakeResponse(429, {"error": {}})

    monkeypatch.setattr(gw.requests, "post", fake_post)
    g = Gateway()
    with pytest.raises(GatewayError, match="не ответила за 5 попыток"):
        g.call("gpt-6-sol", "hi")
    assert calls["n"] == 5
    assert "gpt-6-sol" in g.dead_models  # устойчивый 429 выводит модель из пула


def test_gateway_call_403_marks_dead_and_raises_unavailable(key, monkeypatch):
    resp = FakeResponse(403, {"error": {"message": "geo-blocked"}})
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: resp)
    g = Gateway()
    with pytest.raises(ModelUnavailable, match="403"):
        g.call("gpt-6-sol", "hi")
    assert "gpt-6-sol" in g.dead_models


def test_gateway_call_5xx_retries_then_raises(key, monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **kw):
        calls["n"] += 1
        return FakeResponse(503)

    monkeypatch.setattr(gw.requests, "post", fake_post)
    g = Gateway()
    with pytest.raises(GatewayError):
        g.call("gpt-6-sol", "hi")
    assert calls["n"] == 5


def test_gateway_call_400_bumps_max_tokens_and_retries_to_success(key, monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers, json, timeout):
        calls["n"] += 1
        if json["max_tokens"] < 16:
            return FakeResponse(400, {"error": {"message": "max_tokens must be at least 16"}})
        return _ok_response()

    monkeypatch.setattr(gw.requests, "post", fake_post)
    g = Gateway()
    result = g.call("claude-sonnet-5", "hi", max_tokens=4)  # не gpt_family, не бампится заранее
    assert result.text == "hello"
    assert calls["n"] == 2


def test_gateway_call_400_other_reason_raises_immediately(key, monkeypatch):
    resp = FakeResponse(400, {"error": {"message": "prompt too long"}})
    monkeypatch.setattr(gw.requests, "post", lambda *a, **kw: resp)
    g = Gateway()
    with pytest.raises(GatewayError, match="prompt too long"):
        g.call("claude-sonnet-5", "hi")


def test_gateway_call_network_error_retries_then_raises(key, monkeypatch):
    import requests as real_requests

    calls = {"n": 0}

    def fake_post(*a, **kw):
        calls["n"] += 1
        raise real_requests.ConnectionError("boom")

    monkeypatch.setattr(gw.requests, "post", fake_post)
    g = Gateway()
    with pytest.raises(GatewayError):
        g.call("gpt-6-sol", "hi")
    assert calls["n"] == 5


# ---------- FakeGateway (--dry-run) ----------

def test_fake_gateway_returns_valid_contract_payload():
    fg = FakeGateway()
    result = fg.call("gpt-6-sol", "Ты — агент роя в роли researcher.")
    import json
    payload = json.loads(result.text)
    assert payload["role"] == "researcher"
    assert payload["status"] == "done"
    assert payload["confidence"] == 0.7


def test_fake_gateway_defaults_to_specialist_when_role_unclear():
    fg = FakeGateway()
    result = fg.call("gpt-6-sol", "текст без явной роли")
    import json
    payload = json.loads(result.text)
    assert payload["role"] == "specialist"


def test_fake_gateway_tracks_usage_without_real_cost():
    fg = FakeGateway()
    fg.call("gpt-6-sol", "roleless prompt one")
    fg.call("gpt-6-sol", "roleless prompt two")
    assert fg.usage.calls == 2
    assert fg.usage.actual_cost == 0.0

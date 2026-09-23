"""JSON-контракт агента: извлечение/нормализация/валидация, повтор при браке, фолбэк по моделям."""
from __future__ import annotations

import json

import pytest

from swarm.contract import (
    MAX_TOKENS_CEILING, _extract_json, _normalize, _validate,
    build_prompt, call_agent, call_agent_any,
)
from swarm.gateway import CallResult, GatewayError, ModelUnavailable


def valid_payload(**overrides) -> dict:
    base = {
        "role": "implementer", "task_id": "t1", "status": "done", "confidence": 0.9,
        "result": "готово", "artifacts": [], "tools_used": [], "findings": [],
        "assumptions": [], "self_check": {"goal_met": True, "why": "ok"},
    }
    base.update(overrides)
    return base


# ---------- _extract_json ----------

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_code_block():
    text = "конечно, вот ответ:\n```json\n{\"a\": 1}\n```\nготово"
    assert _extract_json(text) == {"a": 1}


def test_extract_json_preamble_and_postscript_without_fence():
    text = 'вот результат {"a": 1} спасибо'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_malformed_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("не json вообще")


# ---------- _normalize ----------

def test_normalize_findings_strings_become_minor_dicts():
    data = _normalize({"findings": ["что-то не так"]})
    assert data["findings"] == [{"severity": "minor", "what": "что-то не так", "where": "", "fix": ""}]


def test_normalize_findings_mixed_list_keeps_dicts_fixes_strings():
    data = _normalize({"findings": [{"severity": "blocker", "what": "x"}, "строка"]})
    assert data["findings"][0] == {"severity": "blocker", "what": "x"}
    assert data["findings"][1]["severity"] == "minor"


def test_normalize_artifacts_bare_strings_become_create_actions():
    data = _normalize({"artifacts": ["out/x.py"]})
    assert data["artifacts"] == [{"path": "out/x.py", "action": "create", "content": ""}]


def test_normalize_artifacts_missing_becomes_empty_list():
    data = _normalize({})
    assert data["artifacts"] == []


def test_normalize_assumptions_non_list_wrapped():
    data = _normalize({"assumptions": "одно допущение"})
    assert data["assumptions"] == ["одно допущение"]


def test_normalize_self_check_non_dict_becomes_dict():
    data = _normalize({"self_check": "не структура"})
    assert data["self_check"] == {"goal_met": False, "why": "не структура"}


def test_normalize_tools_used_non_list_becomes_empty_list():
    data = _normalize({"tools_used": "read_file"})
    assert data["tools_used"] == []


# ---------- _validate ----------

def test_validate_valid_payload_has_no_problems():
    assert _validate(valid_payload()) == []


def test_validate_missing_required_keys_reported():
    problems = _validate({"status": "done", "result": "x"})
    assert any("нет полей" in p for p in problems)


def test_validate_bad_status_reported():
    problems = _validate(valid_payload(status="finished"))
    assert any("status должен быть" in p for p in problems)


def test_validate_done_with_empty_result_reported():
    problems = _validate(valid_payload(status="done", result=""))
    assert any("result пустой" in p for p in problems)


def test_validate_blocked_without_reason_reported():
    problems = _validate(valid_payload(status="blocked", blocked_reason=""))
    assert any("blocked_reason пустой" in p for p in problems)


def test_validate_confidence_out_of_range_reported():
    problems = _validate(valid_payload(confidence=1.5))
    assert any("confidence вне" in p for p in problems)


def test_validate_high_confidence_with_blockers_is_contradiction():
    problems = _validate(valid_payload(confidence=0.95, findings=[{"severity": "blocker", "what": "x"}]))
    assert any("противоречие" in p for p in problems)


def test_validate_high_confidence_with_only_minor_findings_is_fine():
    problems = _validate(valid_payload(confidence=0.95, findings=[{"severity": "minor", "what": "x"}]))
    assert problems == []


# ---------- build_prompt ----------

def test_build_prompt_scrubs_secrets_for_non_reviewer_roles():
    prompt = build_prompt("implementer", "t1", "используй ключ sk-abcdefghij1234567890")
    assert "sk-abcdefghij1234567890" not in prompt
    assert "<SECRET:API_KEY>" in prompt


def test_build_prompt_does_not_scrub_for_reviewer_checker_acceptor():
    # roles.py: ревьюеры смотрят на уже сгенерированный код, не на пользовательский секрет -
    # скраб там ловил ложноположительные "секреты" (числовые ID, слово password в идентификаторе).
    for role in ("reviewer", "checker", "acceptor"):
        prompt = build_prompt(role, "t1", "some_password_field = 123456789012")
        assert "some_password_field = 123456789012" in prompt


def test_build_prompt_includes_role_and_task_id():
    prompt = build_prompt("researcher", "task-42", "изучи X")
    assert "researcher" in prompt
    assert "task-42" in prompt


def test_build_prompt_acceptor_adds_goal_met_note():
    prompt = build_prompt("acceptor", "t1", "сверь")
    assert "goal_met" in prompt


def test_build_prompt_reviewer_adds_verdict_note():
    prompt = build_prompt("reviewer", "t1", "ревью")
    assert "verdict" in prompt


def test_build_prompt_implementer_adds_compact_note():
    prompt = build_prompt("implementer", "t1", "реализуй")
    assert "КОМПАКТНЫЙ" in prompt.upper() or "компактный" in prompt.lower()


def test_build_prompt_researcher_has_no_compact_note():
    prompt = build_prompt("researcher", "t1", "исследуй")
    assert "компактный" not in prompt.lower()


# ---------- call_agent ----------

class ScriptedGateway:
    """Отдаёт заранее заданную последовательность CallResult/исключений по очереди вызовов."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[tuple[str, str, int]] = []

    def call(self, model, prompt, max_tokens=4096):
        self.calls.append((model, prompt, max_tokens))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _result(text: str, stop_reason: str = "end_turn") -> CallResult:
    return CallResult(text=text, input_tokens=10, output_tokens=10, cost=0.0, stop_reason=stop_reason)


def test_call_agent_happy_path_returns_parsed_reply():
    gw = ScriptedGateway([_result(json.dumps(valid_payload(role="implementer")))])
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X")
    assert reply.status == "done"
    assert reply.result == "готово"
    assert reply.model == "gpt-6-sol"


def test_call_agent_retries_on_malformed_json_then_succeeds():
    gw = ScriptedGateway([
        _result("это не json"),
        _result(json.dumps(valid_payload())),
    ])
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X", max_repairs=2)
    assert reply.status == "done"
    assert len(gw.calls) == 2
    assert "не прошёл проверку" in gw.calls[1][1]


def test_call_agent_exhausts_repairs_returns_synthetic_blocked():
    gw = ScriptedGateway([_result("сломано") for _ in range(3)])  # attempt 0 + 2 repairs
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X", max_repairs=2)
    assert reply.status == "blocked"
    assert reply.raw["spawned_by"] == "contract-repair-exhausted"
    assert len(gw.calls) == 3


def test_call_agent_max_tokens_stop_reason_doubles_budget_and_retries():
    gw = ScriptedGateway([
        _result("", stop_reason="max_tokens"),
        _result(json.dumps(valid_payload())),
    ])
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X", max_tokens=100, max_repairs=2)
    assert reply.status == "done"
    assert gw.calls[1][2] == 200  # удвоенный бюджет на втором вызове


def test_call_agent_max_tokens_doubling_caps_at_ceiling():
    gw = ScriptedGateway([
        _result("", stop_reason="max_tokens"),
        _result(json.dumps(valid_payload())),
    ])
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X",
                        max_tokens=MAX_TOKENS_CEILING - 10, max_repairs=2)
    assert gw.calls[1][2] == MAX_TOKENS_CEILING


def test_call_agent_validation_failure_retries_with_error_explained():
    gw = ScriptedGateway([
        _result(json.dumps(valid_payload(status="bogus"))),
        _result(json.dumps(valid_payload())),
    ])
    reply = call_agent(gw, "gpt-6-sol", "implementer", "t1", "сделай X", max_repairs=2)
    assert reply.status == "done"
    assert "status должен быть" in gw.calls[1][1]


def test_call_agent_retries_when_model_omits_role_then_succeeds_with_role_present():
    # "role" - обязательное поле (REQUIRED_KEYS), так что пропуск роли в ответе браковится
    # _validate() как любой другой изъян контракта и уходит на повтор, а не тихо достраивается
    # через data.setdefault("role", role) - та строка недостижима, пока role в REQUIRED_KEYS.
    missing_role = valid_payload()
    del missing_role["role"]
    gw = ScriptedGateway([
        _result(json.dumps(missing_role)),
        _result(json.dumps(valid_payload(role="researcher"))),
    ])
    reply = call_agent(gw, "gpt-6-sol", "researcher", "t1", "исследуй", max_repairs=2)
    assert reply.status == "done"
    assert reply.raw["role"] == "researcher"
    assert len(gw.calls) == 2
    assert "нет полей" in gw.calls[1][1]


# ---------- call_agent_any ----------

def test_call_agent_any_uses_first_available_model():
    gw = ScriptedGateway([_result(json.dumps(valid_payload()))])
    reply = call_agent_any(gw, ["model-a", "model-b"], "implementer", "t1", "X")
    assert reply.status == "done"
    assert gw.calls[0][0] == "model-a"


def test_call_agent_any_falls_back_when_first_model_unavailable():
    class MultiModelGateway:
        def __init__(self):
            self.seen_models = []

        def call(self, model, prompt, max_tokens=4096):
            self.seen_models.append(model)
            if model == "model-a":
                raise ModelUnavailable("model-a: geo-blocked")
            return _result(json.dumps(valid_payload()))

    gw = MultiModelGateway()
    reply = call_agent_any(gw, ["model-a", "model-b"], "implementer", "t1", "X")
    assert reply.status == "done"
    assert gw.seen_models == ["model-a", "model-b"]


def test_call_agent_any_all_unavailable_returns_synthetic_blocked():
    class AlwaysDownGateway:
        def call(self, model, prompt, max_tokens=4096):
            raise GatewayError(f"{model}: down")

    gw = AlwaysDownGateway()
    reply = call_agent_any(gw, ["model-a", "model-b"], "implementer", "t1", "X")
    assert reply.status == "blocked"
    assert reply.raw["spawned_by"] == "all-models-unavailable"
    assert reply.model == "model-b"

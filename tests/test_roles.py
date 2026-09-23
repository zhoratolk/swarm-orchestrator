"""Менеджер/Диспетчер/Ревизор/Ревью-кворум/Приёмщик — построение решений."""
from __future__ import annotations

import json
import sys

import pytest

from swarm.board import Board, Task
from swarm.gateway import CallResult, GatewayError, ModelUnavailable
from swarm.roles import (
    AUDITOR_FALLBACK_MODEL, AUDITOR_LLM_MODEL, FALLBACK_MODEL, MANAGER_MODEL,
    SpawnAuditor, SpawnRequest,
    _with_fallback, acceptor_check_goal, acceptor_check_ground_truth,
    dispatcher_checker_needed, dispatcher_plan, dispatcher_reviewers_needed,
    manager_decompose, manager_replan, review_quorum_verdict,
)
from swarm.contract import AgentReply


def _result(text: str) -> CallResult:
    return CallResult(text=text, input_tokens=5, output_tokens=5, cost=0.0)


def _contract_text(**overrides) -> str:
    """Полный валидный JSON-контракт (проходит contract._validate REQUIRED_KEYS без правок) -
    все роли-функции в roles.py зовут call_agent_any, который гонит ответ через полный контракт,
    а не принимает голый {"approve_count": ...}/{"verdict": ...} напрямую. Специфичные для роли
    поля (verdict/goal_met - top-level; approve_count/план дispatcher-а - внутри result как своя
    JSON-строка) добавляются через overrides поверх этого скелета."""
    payload = {
        "role": "checker", "task_id": "t1", "status": "done", "confidence": 0.8,
        "result": "", "artifacts": [], "tools_used": [], "findings": [],
        "assumptions": [], "self_check": {"goal_met": True, "why": ""},
    }
    payload.update(overrides)
    return json.dumps(payload)


def _reply(role="checker", **overrides) -> AgentReply:
    raw = {
        "role": role, "task_id": "t1", "status": "done", "confidence": 0.8,
        "result": "ok", "artifacts": [], "tools_used": [], "findings": [],
        "assumptions": [], "self_check": {"goal_met": True, "why": ""},
    }
    raw.update(overrides)
    return AgentReply(raw=raw, model="gpt-6-sol", role_asked=role)


class ScriptedGateway:
    """Один и тот же интерфейс gateway.call(model, prompt, max_tokens), скриптованный ответами
    по очереди вызовов, независимо от того, какая модель спрашивается."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[str] = []
        self.usage = type("U", (), {"actual_cost": 0.0})()

    def call(self, model, prompt, max_tokens=4096):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------- _with_fallback ----------

def test_with_fallback_appends_fallback_when_missing():
    assert _with_fallback("a", "b") == ["a", "b", FALLBACK_MODEL]


def test_with_fallback_dedups_preserving_order():
    assert _with_fallback("a", "a", "b") == ["a", "b", FALLBACK_MODEL]


def test_with_fallback_no_duplicate_when_fallback_already_present():
    assert _with_fallback("a", FALLBACK_MODEL) == ["a", FALLBACK_MODEL]


# ---------- Менеджер: manager_decompose / manager_replan ----------

def test_manager_decompose_parses_task_list_from_llm():
    plan = json.dumps([
        {"id": "t1", "goal": "исследовать X", "deps": [], "kind": "research", "verify_cmd": None},
        {"id": "t2", "goal": "реализовать X", "deps": ["t1"], "kind": "implement", "verify_cmd": "pytest"},
    ])
    gw = ScriptedGateway([_result(_contract_text(role="manager", result=plan))])
    tasks = manager_decompose(gw, "сделать X")
    assert [t.id for t in tasks] == ["t1", "t2"]
    assert tasks[1].deps == ["t1"]
    assert gw.calls[0] == MANAGER_MODEL


def test_manager_decompose_marks_llm_authored_verify_cmd_source():
    # security-гейт: verify_cmd от авто-декомпозиции обязан быть помечен "llm", не "user" -
    # иначе acceptor_check_ground_truth молча выполнит его shell-командой без ревью.
    plan = json.dumps([{"id": "t1", "goal": "x", "deps": [], "kind": "implement", "verify_cmd": "rm -rf /"}])
    gw = ScriptedGateway([_result(_contract_text(role="manager", result=plan))])
    tasks = manager_decompose(gw, "сделать X")
    assert tasks[0].verify_cmd == "rm -rf /"
    assert tasks[0].verify_cmd_source == "llm"


def test_manager_decompose_falls_back_to_single_task_on_bad_json():
    gw = ScriptedGateway([_result(_contract_text(role="manager", result="не json"))])
    tasks = manager_decompose(gw, "сделать X")
    assert len(tasks) == 1
    assert tasks[0].goal == "сделать X"


def test_manager_replan_returns_free_text_decision():
    gw = ScriptedGateway([_result(_contract_text(role="manager", result="переоткрыть t1, сменить подход"))])
    decision = manager_replan(gw, Board.new([Task(id="t1", goal="x")], None), "t1: reject")
    assert decision == "переоткрыть t1, сменить подход"


# ---------- SpawnAuditor ----------

def test_spawn_auditor_rejects_when_cost_cap_reached():
    gw = ScriptedGateway([])
    gw.usage.actual_cost = 1.0
    auditor = SpawnAuditor(gw, cost_cap=1.0)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("implementer", 1, "r", "t1"), task)
    assert verdict.approved_count == 0
    assert "бюджет" in verdict.note


def test_spawn_auditor_rejects_when_role_limit_reached():
    gw = ScriptedGateway([])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x", spawn_count={"implementer": SpawnAuditor.MAX_PER_TASK_PER_ROLE})
    verdict = auditor.review(SpawnRequest("implementer", 1, "r", "t1"), task)
    assert verdict.approved_count == 0
    assert "лимит" in verdict.note


def test_spawn_auditor_approves_small_first_spawn_without_llm_call():
    gw = ScriptedGateway([])  # пустой скрипт - IndexError если реально дойдёт до gateway.call
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("implementer", 2, "r", "t1"), task)
    assert verdict.approved_count == 2
    assert gw.calls == []


def test_spawn_auditor_escalates_borderline_count_to_llm():
    gw = ScriptedGateway([_result(_contract_text(result=json.dumps({"approve_count": 3, "why": "разумно"})))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("implementer", 5, "r", "t1"), task)  # count>3 -> borderline
    assert verdict.approved_count == 3
    assert gw.calls[0] == AUDITOR_LLM_MODEL


def test_spawn_auditor_escalates_specialist_role_to_llm():
    gw = ScriptedGateway([_result(_contract_text(result=json.dumps({"approve_count": 1, "why": "ok"})))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("specialist", 1, "r", "t1"), task)
    assert verdict.approved_count == 1


def test_spawn_auditor_llm_approve_count_clamped_to_requested():
    gw = ScriptedGateway([_result(_contract_text(result=json.dumps({"approve_count": 999, "why": "жадно"})))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("specialist", 2, "r", "t1"), task)
    assert verdict.approved_count == 2  # не больше запрошенного


def test_spawn_auditor_unreadable_verdict_defaults_to_zero():
    # result - валидный контракт, но само поле result не парсится как JSON {"approve_count": ...}
    gw = ScriptedGateway([_result(_contract_text(result="не json вообще"))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("specialist", 1, "r", "t1"), task)
    assert verdict.approved_count == 0


def test_spawn_auditor_approves_by_limit_when_auditor_itself_unavailable():
    # Живой прогон (shakedown-game, 2026-09-24): обе audit-модели легли -> инфраструктурный отказ
    # ("модель недоступна") не то же самое, что содержательное "не обосновано" - иначе легитимный
    # повтор после честного reject душится нулём на пустом месте, застой за 3 итерации гарантирован.
    gw = ScriptedGateway([_result(_contract_text(
        role="checker", status="blocked", result="", blocked_reason="обе audit-модели недоступны",
    ))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    verdict = auditor.review(SpawnRequest("specialist", 2, "r", "t1"), task)
    assert verdict.approved_count == 2  # одобрено по лимитам (count, уже урезанный выше), не 0
    assert "недоступен" in verdict.note


def test_spawn_auditor_borderline_call_uses_auditor_and_its_own_fallback():
    gw = ScriptedGateway([_result(_contract_text(result=json.dumps({"approve_count": 1, "why": "ok"})))])
    auditor = SpawnAuditor(gw, cost_cap=None)
    task = Task(id="t1", goal="x")
    auditor.review(SpawnRequest("specialist", 1, "r", "t1"), task)
    assert gw.calls == [AUDITOR_LLM_MODEL]  # первая модель отвечает, до AUDITOR_FALLBACK_MODEL не доходит


# ---------- dispatcher_plan ----------

def test_dispatcher_plan_research_kind_is_rule_based_no_llm_call():
    gw = ScriptedGateway([])
    task = Task(id="t1", goal="x", kind="research")
    plan = dispatcher_plan(gw, task, Board.new([task], None))
    roles = {r.role for r in plan}
    assert roles == {"researcher", "analyst"}
    assert gw.calls == []


def test_dispatcher_plan_implement_kind_is_rule_based():
    gw = ScriptedGateway([])
    task = Task(id="t1", goal="x", kind="implement")
    plan = dispatcher_plan(gw, task, Board.new([task], None))
    assert len(plan) == 1
    assert plan[0].role == "implementer"


def test_dispatcher_plan_generic_kind_asks_llm_and_parses_response():
    plan_json = json.dumps([{"role": "researcher", "count": 1, "reason": "неясно"}])
    gw = ScriptedGateway([_result(_contract_text(role="dispatcher", result=plan_json))])
    task = Task(id="t1", goal="x", kind="generic")
    plan = dispatcher_plan(gw, task, Board.new([task], None))
    assert plan[0].role == "researcher"
    assert plan[0].count == 1


def test_dispatcher_plan_generic_kind_falls_back_to_specialist_on_bad_json():
    # result - валидный контракт, но само поле result не парсится как JSON-массив плана
    gw = ScriptedGateway([_result(_contract_text(role="dispatcher", result="мусор"))])
    task = Task(id="t1", goal="x", kind="generic")
    plan = dispatcher_plan(gw, task, Board.new([task], None))
    assert len(plan) == 1
    assert plan[0].role == "specialist"


# ---------- dispatcher_reviewers_needed ----------

def test_dispatcher_reviewers_needed_empty_when_no_artifacts():
    replies = [_reply(role="implementer")]
    assert dispatcher_reviewers_needed(Task(id="t1", goal="x"), replies) == []


def test_dispatcher_reviewers_needed_excludes_authors_caps_at_three():
    replies = [AgentReply(raw={"artifacts": [{"path": "a.py"}]}, model="gpt-6-sol", role_asked="implementer")]
    pool = dispatcher_reviewers_needed(Task(id="t1", goal="x"), replies)
    assert "gpt-6-sol" not in pool
    assert len(pool) <= 3


# ---------- review_quorum_verdict ----------

def _review_reply(verdict: str | None, result: str) -> CallResult:
    extra = {"role": "reviewer", "result": result}
    if verdict is not None:
        extra["verdict"] = verdict
    return _result(_contract_text(**extra))


def test_review_quorum_all_approve_yields_approve():
    gw = ScriptedGateway([
        _review_reply("approve", "хорошо"),
        _review_reply("approve", "хорошо"),
        _review_reply("approve", "хорошо"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка",
                                                        ["m1", "m2", "m3"])
    assert verdict == "approve"
    assert len(replies) == 3


def test_review_quorum_any_reject_wins():
    gw = ScriptedGateway([
        _review_reply("approve", "ok"),
        _review_reply("reject", "плохо"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка", ["m1", "m2"])
    assert verdict == "reject"
    assert any("плохо" in r for r in reasons)


def test_review_quorum_approve_with_findings_when_no_reject():
    gw = ScriptedGateway([
        _review_reply("approve", "ok"),
        _review_reply("approve_with_findings", "мелочи"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка", ["m1", "m2"])
    assert verdict == "approve_with_findings"


def test_review_quorum_unreadable_verdict_is_abstain_not_reject():
    # раздел 6: молчащий ревьюер - потерянный голос, не reject. Двойной опрос: обе попытки без verdict.
    gw = ScriptedGateway([
        _review_reply(None, "без verdict"),
        _review_reply(None, "снова без verdict"),
        _review_reply("approve", "ok"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка", ["m1", "m2"])
    assert verdict == "approve"  # единственный читаемый голос - approve, abstain не топит его
    assert any("голос не учтён" in r for r in reasons)


def test_review_quorum_all_abstain_is_reject_not_silent_approve():
    gw = ScriptedGateway([
        _review_reply(None, "молчит"),
        _review_reply(None, "молчит снова"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка", ["m1"])
    assert verdict == "reject"


def test_review_quorum_gives_reviewer_a_second_chance_before_abstain():
    gw = ScriptedGateway([
        _review_reply(None, "нет verdict в первый раз"),
        _review_reply("approve", "теперь есть"),
    ])
    verdict, replies, reasons = review_quorum_verdict(gw, Task(id="t1", goal="x"), "сводка", ["m1"])
    assert verdict == "approve"
    assert len(gw.calls) == 2  # два опроса той же модели, а не сразу abstain


# ---------- dispatcher_checker_needed ----------

def test_checker_needed_single_reply_with_assumptions():
    replies = [_reply(assumptions=["предположил X"])]
    req = dispatcher_checker_needed(Task(id="t1", goal="x"), replies)
    assert req is not None
    assert req.role == "checker"


def test_checker_needed_single_reply_no_assumptions_is_none():
    replies = [_reply(assumptions=[])]
    assert dispatcher_checker_needed(Task(id="t1", goal="x"), replies) is None


def test_checker_needed_diverging_results_triggers_checker():
    replies = [_reply(result="вариант A"), _reply(result="вариант B")]
    req = dispatcher_checker_needed(Task(id="t1", goal="x"), replies)
    assert req is not None


def test_checker_needed_same_result_no_path_conflict_is_none():
    replies = [
        _reply(result="одно и то же", artifacts=[{"path": "a.py"}]),
        _reply(result="одно и то же", artifacts=[{"path": "b.py"}]),
    ]
    assert dispatcher_checker_needed(Task(id="t1", goal="x"), replies) is None


def test_checker_needed_same_path_conflict_triggers_checker():
    replies = [
        _reply(result="одно и то же", artifacts=[{"path": "a.py"}]),
        _reply(result="одно и то же", artifacts=[{"path": "a.py"}]),
    ]
    req = dispatcher_checker_needed(Task(id="t1", goal="x"), replies)
    assert req is not None
    assert "конфликт" in req.reason


# ---------- acceptor_check_goal ----------

def test_acceptor_check_goal_true_when_llm_confirms():
    gw = ScriptedGateway([_result(_contract_text(role="acceptor", goal_met=True, result="да, сделано"))])
    board = Board.new([Task(id="t1", goal="x", status="done")], None)
    ok, why = acceptor_check_goal(gw, "цель", board)
    assert ok is True


def test_acceptor_check_goal_false_when_not_done():
    gw = ScriptedGateway([_result(_contract_text(
        role="acceptor", status="blocked", result="", blocked_reason="не смог",
    ))])
    board = Board.new([Task(id="t1", goal="x")], None)
    ok, why = acceptor_check_goal(gw, "цель", board)
    assert ok is False
    assert "не смог" in why


def test_acceptor_check_goal_false_when_goal_met_not_boolean():
    gw = ScriptedGateway([_result(_contract_text(role="acceptor", goal_met="может быть", result="неясно"))])
    board = Board.new([Task(id="t1", goal="x")], None)
    ok, why = acceptor_check_goal(gw, "цель", board)
    assert ok is False
    assert "понятное goal_met" in why


# ---------- acceptor_check_ground_truth: КЛЮЧЕВОЙ security-гейт ----------

PY = sys.executable
PASS_CMD = f'"{PY}" -c "import sys; sys.exit(0)"'
FAIL_CMD = f'"{PY}" -c "import sys; sys.exit(1)"'


def test_ground_truth_executes_user_verify_cmd_and_passes():
    t = Task(id="t1", goal="x", verify_cmd=PASS_CMD, verify_cmd_source="user")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board)
    assert ok is True
    assert failures == {}
    assert any("ok" in n for n in notes)


def test_ground_truth_executes_user_verify_cmd_and_reports_failure():
    t = Task(id="t1", goal="x", verify_cmd=FAIL_CMD, verify_cmd_source="user")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board)
    assert ok is False
    assert "t1" in failures


def test_ground_truth_skips_llm_verify_cmd_by_default():
    """Дыра: verify_cmd от Менеджера при авто-декомпозиции цели не должен исполняться без
    явного allow_llm_verify_cmd - иначе бесплатная модель без ревью получает shell-доступ."""
    t = Task(id="t1", goal="x", verify_cmd=FAIL_CMD, verify_cmd_source="llm")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board, allow_llm_verify_cmd=False)
    assert ok is True  # пропущенная команда не считается провалом
    assert failures == {}  # и не попадает в failures - не переоткрывает задачу вслепую
    assert any("пропущен" in n for n in notes)


def test_ground_truth_executes_llm_verify_cmd_when_explicitly_allowed():
    t = Task(id="t1", goal="x", verify_cmd=FAIL_CMD, verify_cmd_source="llm")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board, allow_llm_verify_cmd=True)
    assert ok is False
    assert "t1" in failures


def test_ground_truth_user_verify_cmd_ignores_allow_llm_flag():
    """allow_llm_verify_cmd касается ТОЛЬКО verify_cmd_source="llm" - user-заданные команды
    исполняются всегда, флаг их не должен ни включать, ни выключать."""
    t = Task(id="t1", goal="x", verify_cmd=PASS_CMD, verify_cmd_source="user")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board, allow_llm_verify_cmd=False)
    assert ok is True
    assert any("ok" in n for n in notes)  # реально выполнилась, не пропущена


def test_ground_truth_no_verify_cmd_is_silently_skipped():
    t = Task(id="t1", goal="x")
    board = Board.new([t], None)
    ok, notes, failures = acceptor_check_ground_truth(board)
    assert ok is True
    assert notes == []


def test_ground_truth_multiple_tasks_mixed_sources():
    tasks = [
        Task(id="a", goal="x", verify_cmd=PASS_CMD, verify_cmd_source="user"),
        Task(id="b", goal="x", verify_cmd=FAIL_CMD, verify_cmd_source="llm"),
    ]
    board = Board.new(tasks, None)
    ok, notes, failures = acceptor_check_ground_truth(board, allow_llm_verify_cmd=False)
    assert ok is True  # a прошла, b пропущена (не провалена) - обе не считаются провалом
    assert "a" not in failures
    assert "b" not in failures

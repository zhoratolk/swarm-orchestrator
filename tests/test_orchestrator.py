"""Главный цикл: конфиг, repo_context, запись артефактов, контекст зависимостей, волны."""
from __future__ import annotations

import json

import pytest
import yaml

import swarm.orchestrator as orch
from swarm.board import Board, Task
from swarm.contract import AgentReply
from swarm.gateway import CallResult, GatewayError
from swarm.roles import SpawnAuditor, SpawnRequest


# ---------- load_config ----------

def test_load_config_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("goal: сделать X\n", encoding="utf-8")
    cfg = orch.load_config(path)
    assert cfg["goal"] == "сделать X"
    assert cfg["tasks"] == []
    assert cfg["allow_paid"] is False
    assert cfg["allow_llm_verify_cmd"] is False  # дефолт закрытой дыры - не включён сам собой
    assert cfg["cost_cap"] is None
    assert cfg["max_iterations"] == 6
    assert cfg["repo_context"] == []


def test_load_config_explicit_values_override_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "goal: X\nallow_paid: true\nallow_llm_verify_cmd: true\ncost_cap: 2.5\nmax_iterations: 3\n",
        encoding="utf-8",
    )
    cfg = orch.load_config(path)
    assert cfg["allow_paid"] is True
    assert cfg["allow_llm_verify_cmd"] is True
    assert cfg["cost_cap"] == 2.5
    assert cfg["max_iterations"] == 3


def test_load_config_empty_file_still_has_all_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("", encoding="utf-8")
    cfg = orch.load_config(path)
    assert cfg["goal"] == ""
    assert cfg["allow_llm_verify_cmd"] is False


# ---------- build_repo_context ----------

def test_build_repo_context_empty_globs_returns_empty_string(tmp_path):
    assert orch.build_repo_context([], tmp_path) == ""


def test_build_repo_context_reads_matching_files(tmp_path):
    (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / "b.txt").write_text("не питон", encoding="utf-8")
    ctx = orch.build_repo_context(["*.py"], tmp_path)
    assert "a.py" in ctx
    assert "print('a')" in ctx
    assert "b.txt" not in ctx


def test_build_repo_context_recursive_glob(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py") .write_text("x = 1", encoding="utf-8")
    ctx = orch.build_repo_context(["**/*.py"], tmp_path)
    assert "pkg/mod.py" in ctx
    assert "x = 1" in ctx


def test_build_repo_context_skips_undecodable_files(tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01binary junk")
    (tmp_path / "ok.py").write_text("y = 2", encoding="utf-8")
    ctx = orch.build_repo_context(["*"], tmp_path)
    assert "y = 2" in ctx
    assert "bin.dat" not in ctx


def test_build_repo_context_truncates_at_limit_and_warns(tmp_path, monkeypatch, caplog):
    (tmp_path / "big1.py").write_text("a" * 60, encoding="utf-8")
    (tmp_path / "big2.py").write_text("b" * 60, encoding="utf-8")
    monkeypatch.setattr(orch, "REPO_CONTEXT_LIMIT", 100)  # ~81 символ на один блок, ~162 на два

    with caplog.at_level("WARNING"):
        ctx = orch.build_repo_context(["*.py"], tmp_path)

    assert "big1.py" in ctx
    assert "big2.py" not in ctx
    assert any("лимит" in r.message for r in caplog.records)


# ---------- write_artifacts ----------

def test_write_artifacts_creates_file_with_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch.write_artifacts([{"path": "out/x.py", "action": "create", "content": "print(1)"}])
    assert (tmp_path / "out" / "x.py").read_text(encoding="utf-8") == "print(1)"


def test_write_artifacts_skips_empty_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch.write_artifacts([{"path": "out/empty.py", "action": "create", "content": ""}])
    assert not (tmp_path / "out" / "empty.py").exists()


def test_write_artifacts_delete_action_removes_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("stale", encoding="utf-8")
    orch.write_artifacts([{"path": "out/x.py", "action": "delete"}])
    assert not target.exists()


def test_write_artifacts_rejects_path_traversal_outside_cwd(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING"):
        orch.write_artifacts([{"path": "../escape.py", "action": "create", "content": "evil"}])
    assert not (tmp_path.parent / "escape.py").exists()
    assert any("вне рабочей директории" in r.message for r in caplog.records)


def test_write_artifacts_rejects_absolute_path_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "abs_escape.py"
    orch.write_artifacts([{"path": str(outside), "action": "create", "content": "evil"}])
    assert not outside.exists()


def test_write_artifacts_ignores_entries_without_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch.write_artifacts([{"action": "create", "content": "x"}])  # не падает


# ---------- dep_context ----------

def test_dep_context_includes_dependency_artifact_content():
    dep = Task(id="impl", goal="реализовать", artifacts=[{"path": "out/x.py", "content": "def f(): pass"}])
    task = Task(id="tests", goal="протестировать", deps=["impl"])
    board = Board.new([dep, task], None)
    ctx = orch.dep_context(board, task)
    assert "out/x.py" in ctx
    assert "def f(): pass" in ctx


def test_dep_context_includes_longest_research_result_truncated():
    dep = Task(id="research", goal="изучить", results=["короткий вывод", "длинный вывод " * 200])
    task = Task(id="impl", goal="реализовать", deps=["research"])
    board = Board.new([dep, task], None)
    ctx = orch.dep_context(board, task)
    assert "длинный вывод" in ctx
    assert "короткий вывод" not in ctx  # выбран только самый длинный
    assert len(ctx) < len("длинный вывод " * 200) + 200  # обрезан ~1800


def test_dep_context_notes_when_dep_has_nothing():
    dep = Task(id="a", goal="x", status="done")
    task = Task(id="b", goal="y", deps=["a"])
    board = Board.new([dep, task], None)
    ctx = orch.dep_context(board, task)
    assert "не оставила ни файлов, ни текста результата" in ctx


def test_dep_context_missing_dep_id_is_skipped_not_crashed():
    task = Task(id="a", goal="x", deps=["ghost"])
    board = Board.new([task], None)
    assert orch.dep_context(board, task) == ""


# ---------- run_wave ----------

class StubGateway:
    def __init__(self, by_model: dict):
        self.by_model = by_model  # model -> CallResult или Exception
        self.usage = type("U", (), {"actual_cost": 0.0})()
        self.calls_made = 0

    def call(self, model, prompt, max_tokens=4096):
        self.calls_made += 1
        item = self.by_model[model]
        if isinstance(item, Exception):
            raise item
        return item


def _contract(role="implementer", **overrides) -> str:
    payload = {
        "role": role, "task_id": "t1", "status": "done", "confidence": 0.8,
        "result": "готово", "artifacts": [], "tools_used": [], "findings": [],
        "assumptions": [], "self_check": {"goal_met": True, "why": ""},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_run_wave_collects_replies_and_bumps_spawn_count():
    gw = StubGateway({"model-a": CallResult(text=_contract(), input_tokens=1, output_tokens=1, cost=0.0)})
    task = Task(id="t1", goal="x")
    board = Board.new([task], None)
    replies = orch.run_wave(gw, [SpawnRequest("implementer", 1, "r", "t1")], task, board,
                             models_of=lambda role: ["model-a"])
    assert len(replies) == 1
    assert task.spawn_count["implementer"] == 1


def test_run_wave_all_models_down_yields_blocked_reply_not_a_dropped_wave():
    # call_agent_any уже сама деградирует до синтетического blocked, если ВСЕ модели (включая
    # встроенный фолбэк [model, FALLBACK_MODEL], который run_wave подставляет сам) недоступны -
    # run_wave должен пронести этот ответ дальше как обычную реплику волны, а не потерять её.
    from swarm.roles import FALLBACK_MODEL
    gw = StubGateway({
        "model-a": GatewayError("down"),
        FALLBACK_MODEL: GatewayError("down too"),
    })
    task = Task(id="t1", goal="x")
    board = Board.new([task], None)
    replies = orch.run_wave(gw, [SpawnRequest("implementer", 1, "r", "t1")], task, board,
                             models_of=lambda role: ["model-a"])
    assert len(replies) == 1
    assert replies[0].status == "blocked"


def test_run_wave_empty_requests_returns_empty_list():
    gw = StubGateway({})
    task = Task(id="t1", goal="x")
    board = Board.new([task], None)
    assert orch.run_wave(gw, [], task, board, models_of=lambda role: ["m"]) == []


# ---------- process_task: state-machine логика волны/ревью/приёмки одной задачи ----------

def test_process_task_auditor_rejects_everything_blocks_task():
    gw = StubGateway({})  # бюджет исчерпан заранее - до gateway.call дело не доходит вовсе
    task = Task(id="t1", goal="x", kind="implement")
    board = Board.new([task], None)
    auditor = SpawnAuditor(gw, cost_cap=0.0)
    orch.process_task(gw, auditor, task, board, allow_paid=False)
    assert task.status == "blocked"
    assert any("ревизор не одобрил" in f["what"] for f in task.findings)
    assert gw.calls_made == 0


def test_process_task_happy_path_single_implementer_no_artifacts_is_done():
    gw = StubGateway({"gpt-6-sol": CallResult(
        text=_contract(role="implementer", status="done", result="сделано"),
        input_tokens=1, output_tokens=1, cost=0.0,
    )})
    task = Task(id="t1", goal="x", kind="implement")
    board = Board.new([task], None)
    auditor = SpawnAuditor(gw, cost_cap=None)
    orch.process_task(gw, auditor, task, board, allow_paid=False)
    assert task.status == "done"
    assert task.results == ["сделано"]


def test_process_task_reviewer_quorum_reject_blocks_and_records_feedback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    impl_reply = _contract(
        role="implementer", status="done", result="сделано",
        artifacts=[{"path": "out/x.py", "action": "create", "content": "code"}],
    )
    reject_reply = json.dumps({
        "role": "reviewer", "task_id": "t1", "status": "done", "confidence": 0.8,
        "result": "плохой код", "artifacts": [], "tools_used": [], "findings": [],
        "assumptions": [], "self_check": {"goal_met": False, "why": ""}, "verdict": "reject",
    })
    # автор implementer'а - gpt-6-sol, REVIEWER_POOL минус автор -> первые 3 из
    # [gpt-6-luna, gpt-5.6-luna, claude-opus-5.5]
    gw = StubGateway({
        "gpt-6-sol": CallResult(text=impl_reply, input_tokens=1, output_tokens=1, cost=0.0),
        "gpt-6-luna": CallResult(text=reject_reply, input_tokens=1, output_tokens=1, cost=0.0),
        "gpt-5.6-luna": CallResult(text=reject_reply, input_tokens=1, output_tokens=1, cost=0.0),
        "claude-opus-5.5": CallResult(text=reject_reply, input_tokens=1, output_tokens=1, cost=0.0),
    })
    task = Task(id="t1", goal="x", kind="implement")
    board = Board.new([task], None)
    auditor = SpawnAuditor(gw, cost_cap=None)
    orch.process_task(gw, auditor, task, board, allow_paid=False)
    assert task.status == "blocked"
    assert task.feedback  # причина reject переживает переоткрытие задачи, не теряется
    assert any("ревью отклонило" in f["what"] for f in task.findings)


def test_process_task_needs_info_blocks_with_feedback():
    needs_info_reply = json.dumps({
        "role": "implementer", "task_id": "t1", "status": "needs_info", "confidence": 0.5,
        "result": "нужна спецификация формата вывода", "artifacts": [], "tools_used": [],
        "findings": [], "assumptions": [], "self_check": {"goal_met": False, "why": "не хватает информации"},
    })
    gw = StubGateway({"gpt-6-sol": CallResult(
        text=needs_info_reply, input_tokens=1, output_tokens=1, cost=0.0,
    )})
    task = Task(id="t1", goal="x", kind="implement")
    board = Board.new([task], None)
    auditor = SpawnAuditor(gw, cost_cap=None)
    orch.process_task(gw, auditor, task, board, allow_paid=False)
    assert task.status == "blocked"
    assert any("needs_info" in f["what"] for f in task.findings)
    assert task.feedback

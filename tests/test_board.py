"""Доска задач: сериализация, зависимости, обратная совместимость board.json."""
from __future__ import annotations

import json

from swarm.board import Board, Task


def test_task_brief_returns_goal_when_no_feedback():
    t = Task(id="t1", goal="сделать X")
    assert t.brief() == "сделать X"


def test_task_brief_appends_last_three_feedback_entries():
    t = Task(id="t1", goal="сделать X", feedback=["f1", "f2", "f3", "f4"])
    brief = t.brief()
    assert "сделать X" in brief
    assert "- f2" in brief and "- f3" in brief and "- f4" in brief
    assert "- f1" not in brief  # только последние 3


def test_task_log_appends_timestamped_event():
    t = Task(id="t1", goal="x")
    t.log("dispatch", role="researcher")
    assert len(t.history) == 1
    assert t.history[0]["event"] == "dispatch"
    assert t.history[0]["role"] == "researcher"
    assert "t" in t.history[0]


def test_task_verify_cmd_source_defaults_to_user():
    t = Task(id="t1", goal="x", verify_cmd="echo hi")
    assert t.verify_cmd_source == "user"


def test_board_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "board.json"
    tasks = [
        Task(id="a", goal="A", deps=[]),
        Task(id="b", goal="B", deps=["a"], verify_cmd="echo hi", verify_cmd_source="llm"),
    ]
    board = Board.new(tasks, path)
    board.save()

    assert path.exists()
    loaded = Board.load(path)
    assert set(loaded.tasks) == {"a", "b"}
    assert loaded.tasks["b"].deps == ["a"]
    assert loaded.tasks["b"].verify_cmd_source == "llm"


def test_board_load_tolerates_missing_verify_cmd_source_field(tmp_path):
    """board.json написанный ДО появления verify_cmd_source (старый прогон/резюме) не должен
    падать на Task(**t) — отсутствующее поле обязано взять дефолт датакласса."""
    path = tmp_path / "board.json"
    old_style = {
        "a": {"id": "a", "goal": "A", "deps": [], "verify_cmd": "echo hi", "kind": "generic",
              "status": "queued", "history": [], "artifacts": [], "findings": [],
              "spawn_count": {}, "feedback": [], "results": []},
    }
    path.write_text(json.dumps(old_style), encoding="utf-8")

    board = Board.load(path)
    assert board.tasks["a"].verify_cmd_source == "user"  # дефолт, не падение


def test_board_ready_only_when_all_deps_done():
    a = Task(id="a", goal="A", status="done")
    b = Task(id="b", goal="B", deps=["a"], status="queued")
    c = Task(id="c", goal="C", deps=["a", "b"], status="queued")
    board = Board.new([a, b, c], None)

    ready_ids = {t.id for t in board.ready()}
    assert ready_ids == {"b"}  # c зависит от b, которая ещё не done


def test_board_ready_excludes_non_queued_tasks():
    a = Task(id="a", goal="A", status="in_progress")
    board = Board.new([a], None)
    assert board.ready() == []


def test_board_pending_excludes_only_done():
    tasks = [
        Task(id="a", goal="A", status="done"),
        Task(id="b", goal="B", status="blocked"),
        Task(id="c", goal="C", status="queued"),
    ]
    board = Board.new(tasks, None)
    assert {t.id for t in board.pending()} == {"b", "c"}


def test_board_blockers_filters_by_severity():
    a = Task(id="a", goal="A", findings=[
        {"severity": "blocker", "what": "x"},
        {"severity": "minor", "what": "y"},
    ])
    board = Board.new([a], None)
    blockers = board.blockers()
    assert len(blockers) == 1
    assert blockers[0]["what"] == "x"


def test_board_all_done_true_only_when_every_task_done():
    board = Board.new([Task(id="a", goal="A", status="done")], None)
    assert board.all_done() is True
    board2 = Board.new([Task(id="a", goal="A", status="done"),
                         Task(id="b", goal="B", status="queued")], None)
    assert board2.all_done() is False


def test_board_any_permanently_blocked():
    board = Board.new([Task(id="a", goal="A", status="blocked")], None)
    assert board.any_permanently_blocked() is True
    board2 = Board.new([Task(id="a", goal="A", status="queued")], None)
    assert board2.any_permanently_blocked() is False

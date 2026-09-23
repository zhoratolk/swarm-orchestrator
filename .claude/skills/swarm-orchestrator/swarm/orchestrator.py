"""Главный цикл (раздел 7 протокола) и CLI. Запуск: python -m swarm.orchestrator run tasks.yaml [--dry-run] [--resume]."""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .board import Board, Task
from .contract import call_agent, AgentReply
from .gateway import Gateway, FakeGateway, Usage, GatewayError, ModelUnavailable
from .roles import (
    manager_decompose, manager_replan, dispatcher_plan, dispatcher_checker_needed,
    SpawnAuditor, SpawnRequest, acceptor_check_goal, acceptor_check_ground_truth,
    ROLE_MODEL_DEFAULTS,
)

log = logging.getLogger("swarm")


def setup_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
    ])


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("goal", "")
    cfg.setdefault("tasks", [])
    cfg.setdefault("allow_paid", False)
    cfg.setdefault("cost_cap", None)
    cfg.setdefault("max_iterations", 6)
    return cfg


def run_wave(gateway, requests_: list[SpawnRequest], task: Task, models_of, workers: int = 6) -> list[AgentReply]:
    calls = []
    for req in requests_:
        models = models_of(req.role)
        for i in range(req.count):
            model = models[i % len(models)]
            calls.append((req.role, model))
            task.spawn_count[req.role] = task.spawn_count.get(req.role, 0) + 1

    if not calls:
        return []

    replies: list[AgentReply] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(calls))) as pool:
        futures = {
            pool.submit(call_agent, gateway, model, role, task.id, task.goal): (role, model)
            for role, model in calls
        }
        for fut in as_completed(futures):
            role, model = futures[fut]
            try:
                replies.append(fut.result())
            except ModelUnavailable as e:
                log.warning("модель %s недоступна для роли %s: %s — пропуск", model, role, e)
            except GatewayError as e:
                log.error("сбой вызова %s/%s: %s", role, model, e)
    return replies


def process_task(gateway, auditor: SpawnAuditor, task: Task, board: Board,
                  allow_paid: bool) -> None:
    def models_of(role: str) -> list[str]:
        models = ROLE_MODEL_DEFAULTS.get(role, ["gpt-6-sol"])
        if not allow_paid:
            from .gateway import MODELS
            free = [m for m in models if MODELS.get(m, {}).get("free", True)]
            models = free or models
        return models

    task.status = "dispatched"
    task.log("dispatch")
    plan = dispatcher_plan(gateway, task, board)
    approved: list[SpawnRequest] = []
    for req in plan:
        verdict = auditor.review(req, task)
        task.log("audit", role=req.role, requested=req.count, approved=verdict.approved_count, note=verdict.note)
        if verdict.approved_count > 0:
            approved.append(SpawnRequest(req.role, verdict.approved_count, req.reason, task.id))
        else:
            log.info("ревизор отклонил спавн %s x%d на %s: %s", req.role, req.count, task.id, verdict.note)

    if not approved:
        task.status = "blocked"
        task.findings.append({"severity": "blocker", "what": "ревизор не одобрил ни один спавн",
                               "where": task.id, "fix": "пересмотреть задачу или лимиты ревизора"})
        return

    task.status = "in_progress"
    replies = run_wave(gateway, approved, task, models_of)
    if not replies:
        task.status = "blocked"
        task.findings.append({"severity": "blocker", "what": "волна не вернула ни одного ответа",
                               "where": task.id, "fix": "проверить доступность моделей"})
        return

    for r in replies:
        task.artifacts += r.artifacts
        task.findings += r.findings
        task.log("reply", role=r.role_asked, model=r.model, status=r.status, confidence=r.confidence)

    checker_req = dispatcher_checker_needed(task, replies)
    if checker_req:
        verdict = auditor.review(checker_req, task)
        task.log("audit", role="checker", requested=checker_req.count,
                  approved=verdict.approved_count, note=verdict.note)
        if verdict.approved_count > 0:
            checker_replies = run_wave(gateway, [SpawnRequest("checker", verdict.approved_count,
                                                                checker_req.reason, task.id)],
                                        task, models_of)
            for r in checker_replies:
                task.findings += r.findings
                task.log("checker_reply", model=r.model, status=r.status)

    blockers = [f for f in task.findings if f.get("severity") == "blocker"]
    if blockers:
        task.status = "blocked"
    elif any(r.status == "done" for r in replies):
        task.status = "done"
    else:
        task.status = "needs_review"


def run(config_path: Path, run_dir: Path, dry_run: bool, resume: bool):
    setup_logging(run_dir)
    cfg = load_config(config_path)

    usage = Usage()
    gateway = FakeGateway(usage) if dry_run else Gateway(usage)
    if dry_run:
        log.warning("--dry-run: реальные вызовы шлюза НЕ делаются, ответы — заглушки")

    board_path = run_dir / "board.json"
    if resume and board_path.exists():
        board = Board.load(board_path)
        log.info("резюме прогона из %s", board_path)
    elif cfg["tasks"]:
        tasks = [Task(id=t["id"], goal=t["goal"], deps=t.get("deps", []),
                       kind=t.get("kind", "generic"), verify_cmd=t.get("verify_cmd"))
                 for t in cfg["tasks"]]
        board = Board.new(tasks, board_path)
    else:
        log.info("задач в конфиге нет — Менеджер раскладывает цель сам")
        tasks = manager_decompose(gateway, cfg["goal"])
        board = Board.new(tasks, board_path)
    board.save()

    auditor = SpawnAuditor(gateway, cfg["cost_cap"])
    stagnant_rounds = 0
    prev_blocker_count = None

    for iteration in range(1, cfg["max_iterations"] + 1):
        log.info("=== итерация %d ===", iteration)
        ready = board.ready()
        if not ready and not board.pending():
            break
        if not ready:
            log.warning("нет готовых задач, но есть незакрытые (%d) — вероятно, все blocked",
                        len(board.pending()))
            break

        for task in ready:
            log.info("задача %s: %s", task.id, task.goal[:120])
            process_task(gateway, auditor, task, board, cfg["allow_paid"])
            board.save()

        blockers = board.blockers()
        if prev_blocker_count is not None and len(blockers) >= prev_blocker_count:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        prev_blocker_count = len(blockers)

        if stagnant_rounds >= 3:
            log.error("застой: 3 итерации подряд без сокращения блокеров (%d) — эскалация человеку",
                      len(blockers))
            break

        if cfg["cost_cap"] is not None and usage.actual_cost >= cfg["cost_cap"]:
            log.error("бюджет исчерпан: $%.4f >= $%.4f — остановка", usage.actual_cost, cfg["cost_cap"])
            break

        if blockers and board.any_permanently_blocked():
            summary = "; ".join(f"{b.get('what')} ({b.get('where')})" for b in blockers[:5])
            decision = manager_replan(gateway, board, summary)
            log.info("менеджер: %s", decision)
            for t in board.tasks.values():
                if t.status == "blocked":
                    t.status = "queued"
                    t.findings = []
            board.save()

        if board.all_done():
            break

    acceptance_notes = []
    goal_met, why = False, "цикл не дошёл до приёмки"
    ground_truth_ok, gt_notes = True, []
    if board.all_done():
        goal_met, why = acceptor_check_goal(gateway, cfg["goal"], board)
        ground_truth_ok, gt_notes = acceptor_check_ground_truth(board)
        acceptance_notes.append(f"цель: {'ok' if goal_met else 'НЕ ok'} — {why}")
        acceptance_notes += gt_notes

    board.save()
    write_report(run_dir, cfg, board, usage, goal_met and ground_truth_ok, acceptance_notes)
    log.info("готово: %s. Отчёт — %s", "УСПЕХ" if (goal_met and ground_truth_ok) else "не принято",
              run_dir / "report.md")


def write_report(run_dir: Path, cfg: dict, board: Board, usage: Usage, success: bool, notes: list[str]):
    lines = [
        f"# Отчёт роя", "",
        f"**Цель:** {cfg['goal']}", "",
        f"**Итог:** {'УСПЕХ' if success else 'НЕ ПРИНЯТО'}", "",
        "## Приёмка", *[f"- {n}" for n in notes], "",
        "## Задачи",
    ]
    for t in board.tasks.values():
        lines.append(f"- **{t.id}** [{t.status}] — {t.goal[:100]}")
        for f in t.findings:
            lines.append(f"  - {f.get('severity')}: {f.get('what')}")
    lines += ["", "## Расход токенов и стоимость", "", usage.report()]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    load_dotenv()
    p = argparse.ArgumentParser(prog="swarm-orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="запустить рой на файле задачи")
    r.add_argument("config", type=Path, help="YAML с целью и/или списком задач")
    r.add_argument("--run-dir", type=Path, default=Path("run"), help="куда писать board.json/run.log/report.md")
    r.add_argument("--dry-run", action="store_true", help="без ключа и сети, заглушка вместо шлюза")
    r.add_argument("--resume", action="store_true", help="продолжить с board.json в --run-dir")
    args = p.parse_args()

    if args.cmd == "run":
        run(args.config, args.run_dir, args.dry_run, args.resume)


if __name__ == "__main__":
    main()

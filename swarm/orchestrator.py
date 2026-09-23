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
from .contract import call_agent_any, AgentReply
from .gateway import Gateway, FakeGateway, Usage, GatewayError, ModelUnavailable
from .roles import (
    manager_decompose, manager_replan, dispatcher_plan, dispatcher_checker_needed,
    dispatcher_reviewers_needed, review_quorum_verdict,
    SpawnAuditor, SpawnRequest, acceptor_check_goal, acceptor_check_ground_truth,
    ROLE_MODEL_DEFAULTS, FALLBACK_MODEL,
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


def write_artifacts(artifacts: list[dict]) -> None:
    """Реально кладёт то, что агент заявил в artifacts[], на диск (относительно текущей директории).

    Раньше этого шага не было вообще: агенты честно писали содержимое файла в JSON, ревьюеры его видели
    (dep_context/сводка для ревью читают task.artifacts в памяти), а на диске ничего не появлялось — из-за
    этого verify_cmd падал на КАЖДОМ прогоне, независимо от качества кода. Путь с .. или абсолютный —
    пропускаем, не вылезаем за пределы рабочей директории."""
    cwd = Path.cwd().resolve()
    for a in artifacts:
        path = a.get("path")
        if not path:
            continue
        dest = (cwd / path).resolve()
        if cwd not in dest.parents and dest != cwd:
            log.warning("артефакт вне рабочей директории пропущен: %s", path)
            continue
        action = a.get("action", "create")
        if action == "delete":
            if dest.exists():
                dest.unlink()
            continue
        content = a.get("content", "")
        if not content:
            continue  # путь без содержимого — нечего писать, не создаём пустышку
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        log.info("артефакт записан: %s (%d байт)", dest.relative_to(cwd), len(content))


def dep_context(board: Board, task: Task) -> str:
    """Что реально сделали задачи-зависимости — путь и содержимое файлов, не только их цель.

    Раньше зависимая задача видела только текст своей собственной цели: "tests" зависит от "implement",
    но не получала ни пути файла, ни сигнатуры функции, ни того, что implement реально написал — только
    свою же фразу из YAML. Реализатор честно отвечал needs_info, спрашивая то, что уже есть на доске."""
    parts = []
    for dep_id in task.deps:
        dt = board.tasks.get(dep_id)
        if not dt:
            continue
        for a in dt.artifacts:
            if a.get("content"):
                parts.append(f"Из задачи \"{dep_id}\" ({dep_id}), файл {a.get('path')}:\n{a['content'][:3000]}")
        # research/analysis-задачи ничего не пишут на диск — их результат ТОЛЬКО текст result.
        # Раньше при пустых artifacts сюда падала бесполезная заглушка "см. history", а сам result
        # нигде не сохранялся вообще — реализатор честно не мог узнать выводы research.
        # Только ОДИН, самый длинный (обычно самый содержательный) ответ, коротко обрезанный — три
        # полных эссе исследователей (по 3000 символов каждое) раздували промпт implementer'а так, что
        # маленькие бесплатные модели с небольшим общим окном контекста молча резали СВОЙ вывод, независимо
        # от выставленного max_tokens. Экономия контекста тут важнее полноты.
        if dt.results:
            best = max(dt.results, key=len)
            parts.append(f"Из задачи \"{dep_id}\" ({dep_id}), выводы:\n{best[:1800]}")
        if not dt.artifacts and not dt.results:
            parts.append(f"Задача \"{dep_id}\" отмечена done, но не оставила ни файлов, ни текста результата.")
    return "\n\n".join(parts)



# artifacts[].content везёт полный текст файла ВНУТРИ экранированной JSON-строки (кавычки,
# переводы строк как \n) — 4096 дефолтных токенов хватает на рассуждение и findings, но не на
# файл в несколько КБ поверх них. Три живых прогона подряд обрывали implement/checker на этом
# ровно там, где менеджер сам верно диагностировал «обрыв по лимиту», а лимит никто не поднимал.
ROLE_MAX_TOKENS = {"implementer": 8192, "checker": 8192, "specialist": 8192}


def run_wave(gateway, requests_: list[SpawnRequest], task: Task, board: Board, models_of,
             workers: int = 6) -> list[AgentReply]:
    extra = dep_context(board, task)
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
        # у каждого вызова свой фолбэк (раздел 11) — недоступность одной модели не съедает волну
        futures = {
            pool.submit(call_agent_any, gateway, [model, FALLBACK_MODEL] if model != FALLBACK_MODEL else [model],
                        role, task.id, task.brief(), extra,
                        ROLE_MAX_TOKENS.get(role, 4096)): (role, model)
            for role, model in calls
        }
        for fut in as_completed(futures):
            role, model = futures[fut]
            try:
                replies.append(fut.result())
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
    replies = run_wave(gateway, approved, task, board, models_of)
    if not replies:
        task.status = "blocked"
        task.findings.append({"severity": "blocker", "what": "волна не вернула ни одного ответа",
                               "where": task.id, "fix": "проверить доступность моделей"})
        return

    for r in replies:
        task.artifacts += r.artifacts
        task.findings += r.findings
        if r.status == "done" and r.result:
            task.results.append(r.result)
        task.log("reply", role=r.role_asked, model=r.model, status=r.status, confidence=r.confidence)
    for r in replies:
        write_artifacts(r.artifacts)

    checker_req = dispatcher_checker_needed(task, replies)
    if checker_req:
        verdict = auditor.review(checker_req, task)
        task.log("audit", role="checker", requested=checker_req.count,
                  approved=verdict.approved_count, note=verdict.note)
        if verdict.approved_count > 0:
            checker_replies = run_wave(gateway, [SpawnRequest("checker", verdict.approved_count,
                                                                checker_req.reason, task.id)],
                                        task, board, models_of)
            for r in checker_replies:
                task.findings += r.findings
                task.log("checker_reply", model=r.model, status=r.status)

    blockers = [f for f in task.findings if f.get("severity") == "blocker"]
    if blockers:
        task.status = "blocked"
        return

    reviewer_models = dispatcher_reviewers_needed(task, replies)
    if reviewer_models:
        review_req = SpawnRequest("reviewer", len(reviewer_models), "кворум ревью готового артефакта", task.id)
        verdict = auditor.review(review_req, task)
        task.log("audit", role="reviewer", requested=review_req.count,
                  approved=verdict.approved_count, note=verdict.note)
        if verdict.approved_count >= 1:
            # ревьюерам — реальное содержимое файлов, не только пересказ, иначе им физически нечего оценивать
            parts = [r.result for r in replies if r.result]
            for r in replies:
                for a in r.artifacts:
                    if a.get("content"):
                        parts.append(f"--- {a.get('path')} ---\n{a['content'][:4000]}")
            summary = "\n\n".join(parts)
            final_verdict, review_replies, reasons = review_quorum_verdict(
                gateway, task, summary, reviewer_models[:verdict.approved_count])
            # models в логе — кто РЕАЛЬНО ответил (после фолбэка), не то, что запрашивали
            task.log("review", verdict=final_verdict, models=[r.model for r in review_replies], reasons=reasons)
            if final_verdict == "reject":
                task.status = "blocked"
                reason_text = "; ".join(reasons) or "ревьюеры не объяснили reject"
                task.findings.append({
                    "severity": "blocker", "what": "ревью отклонило артефакт (кворум reject)",
                    "where": task.id, "fix": reason_text,
                })
                # это ГЛАВНАЯ причина застоя раньше: причину отклонения теряли при переоткрытии задачи —
                # исполнитель на следующем заходе писал то же самое вслепую. feedback переживает reset findings.
                task.feedback.append(f"ревью отклонило: {reason_text}")
                return
            if final_verdict == "approve_with_findings":
                for r in review_replies:
                    task.findings += [f for f in r.findings if f.get("severity") != "blocker"]

    if any(r.status == "done" for r in replies):
        task.status = "done"
        return

    # needs_info от исполнителя раньше зависал тупиком: нет findings -> replan-ветка его не видела,
    # ready() тоже не подбирает needs_review -> задача молча стояла до конца прогона. Заводим через
    # тот же blocked-путь, что и reject, с тем же feedback-каналом, чтобы Реализатор на новом заходе
    # увидел, чего именно не хватило, а не повторял то же самое вслепую.
    asks = [r.result for r in replies if r.status == "needs_info" and r.result]
    task.status = "blocked"
    task.findings.append({
        "severity": "blocker", "what": "исполнитель просит уточнение (needs_info)",
        "where": task.id, "fix": "; ".join(asks) or "см. reply в history",
    })
    if asks:
        task.feedback.append("реализатору не хватило: " + "; ".join(asks))


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
            # раньше сюда шли только what/where — reject reject reject без сути; менеджер честно отвечал
            # "недостаточно контекста" и цикл топтался. Теперь fix/причина едет тоже.
            summary = "; ".join(f"{b.get('what')} ({b.get('where')}): {b.get('fix', '')}" for b in blockers[:5])
            decision = manager_replan(gateway, board, summary)
            log.info("менеджер: %s", decision)
            for t in board.tasks.values():
                if t.status == "blocked":
                    t.status = "queued"
                    t.findings = []  # findings — временный разбор этого захода, feedback остаётся и едет дальше
            board.save()

        if board.all_done():
            # Раньше приёмка (реальный verify_cmd) шла ПОСЛЕ выхода из цикла — провал ground truth
            # заканчивал прогон отчётом "не принято" без единого шанса переоткрыть проваленную задачу,
            # хотя ревью каждую задачу формально одобрило. По протоколу цикл должен крутиться до
            # успеха/доказанного тупика/застоя, а не сдаваться на первом расхождении ревью с реальностью.
            ground_truth_ok, gt_notes, gt_failures = acceptor_check_ground_truth(board)
            if not ground_truth_ok:
                for tid, detail in gt_failures.items():
                    t = board.tasks.get(tid)
                    if not t:
                        continue
                    t.status = "queued"
                    t.findings = []
                    t.feedback.append(f"verify_cmd провалился на реальном прогоне: {detail}")
                    log.info("приёмка: verify_cmd провалил %s, переоткрываю с реальной ошибкой", tid)
                board.save()
                continue
            break

    acceptance_notes = []
    goal_met, why = False, "цикл не дошёл до приёмки"
    ground_truth_ok, gt_notes = True, []
    if board.all_done():
        goal_met, why = acceptor_check_goal(gateway, cfg["goal"], board)
        ground_truth_ok, gt_notes, _ = acceptor_check_ground_truth(board)
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

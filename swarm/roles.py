"""Менеджер, Диспетчер, Ревизор спавна, Приёмщик — построение решений поверх contract.call_agent.

Менеджер и Диспетчер сами один раз зовут LLM за решением (раздел 4). Ревизор в основном
правила (дёшево, детерминизм), LLM зовёт только на пограничные случаи — раздел 4 требует
именно так: «следить, чтобы спавнер создавал агентов только когда реально надо».
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

from .board import Board, Task
from .contract import call_agent, call_agent_any, AgentReply

MANAGER_MODEL = "claude-opus-5.5"
DISPATCHER_MODEL = "gpt-6-luna"
AUDITOR_LLM_MODEL = "nemotron-3-ultra-550b-a55b"
# У AUDITOR_LLM_MODEL нет собственного фолбэка отдельно от общего FALLBACK_MODEL — а это одна и та же
# модель. Живой прогон: nemotron ловит устойчивый 429 (её же используют как фолбэк ВСЕ роли, поэтому
# она перегружена чаще прочих), выпадает из пула, и Ревизор спавна остаётся вообще без модели —
# каждый пограничный спавн (в т.ч. законный повтор после честного reject) молча отклоняется с
# "все модели недоступны", застой гарантирован. Второй, РЕАЛЬНО другой провайдер как подстраховка.
AUDITOR_FALLBACK_MODEL = "gpt-5.6-luna"
ACCEPTOR_MODELS = ("claude-opus-5.5", "gpt-6-sol")

# Раздел 11: единственная модель без геоблоков на некоторых сетях (проверено эмпирически —
# Anthropic/OpenAI-семейства отдают 403 permission_error с рядом IP, NVIDIA-инфра нет). Если
# основная модель роли недоступна в этом прогоне, следующий вызов той же роли уходит сюда —
# рой не падает целиком из-за одного заблокированного провайдера.
FALLBACK_MODEL = "nemotron-3-ultra-550b-a55b"

ROLE_MODEL_DEFAULTS = {
    "researcher": ["gpt-6-sol", "gpt-6-luna", FALLBACK_MODEL],
    "analyst": ["gpt-5.6-luna", FALLBACK_MODEL],
    "implementer": ["gpt-6-sol", "claude-opus-5.5", FALLBACK_MODEL],
    "checker": ["gpt-6-luna", FALLBACK_MODEL],
    "specialist": ["gpt-6-sol", FALLBACK_MODEL],
}


def _with_fallback(*models: str) -> list[str]:
    out = list(dict.fromkeys(models))  # без дублей, порядок сохранён
    if FALLBACK_MODEL not in out:
        out.append(FALLBACK_MODEL)
    return out


# ---------- Менеджер ----------

def manager_decompose(gateway, goal: str, repo_context: str = "", max_tokens: int = 4096) -> list[Task]:
    """Раскладывает цель на задачи один раз в начале, если задачи не заданы в YAML руками.

    repo_context — исходники целевого репозитория (cfg["repo_context"] в YAML), если заданы. Без
    этого Менеджер планирует задачи вслепую, не видя реального кода — годится для задач "написать
    с нуля", но бессмысленно для "найти и исправить баг в существующем коде"."""
    brief = (
        f"Разложи цель на независимые (по возможности) задачи для роя. Цель:\n{goal}\n\n"
        "В result верни JSON-массив задач строкой: "
        '[{"id": "t1", "goal": "...", "deps": [], "kind": "research|implement|generic", "verify_cmd": null}, ...]. '
        "id короткие, deps — id других задач из этого же списка, kind подсказывает какого рода работа."
    )
    reply = call_agent_any(gateway, _with_fallback(MANAGER_MODEL), "manager", "decompose", brief,
                            repo_context, max_tokens=max_tokens)
    tasks: list[Task] = []
    if reply.status == "done":
        try:
            raw = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
            for t in raw:
                tasks.append(Task(id=t["id"], goal=t["goal"], deps=t.get("deps", []),
                                   kind=t.get("kind", "generic"), verify_cmd=t.get("verify_cmd"),
                                   verify_cmd_source="llm"))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if not tasks:
        tasks = [Task(id="t1", goal=goal, deps=[], kind="generic")]
    return tasks


def manager_replan(gateway, board: Board, findings_summary: str, repo_context: str = "",
                    max_tokens: int = 2048) -> str:
    """Вызывается при застое/reject — Менеджер решает, что делать дальше. Возвращает свободный текст-решение."""
    brief = (
        f"Доска задач застряла. Находки: {findings_summary}\n"
        "Реши: какую задачу переоткрыть, что изменить в подходе, или что эскалировать человеку. "
        "Кратко в result."
    )
    reply = call_agent_any(gateway, _with_fallback(MANAGER_MODEL), "manager", "replan", brief,
                            repo_context, max_tokens=max_tokens)
    return reply.result


# ---------- Ревизор спавна ----------

@dataclass
class SpawnRequest:
    role: str
    count: int
    reason: str
    task_id: str


@dataclass
class AuditVerdict:
    approved_count: int
    note: str


class SpawnAuditor:
    """Правила прежде всего (раздел 4): максимум параллельных агентов на задачу, запрет пустого
    повтора роли, бюджетный потолок. На пограничные случаи (N>3, повторный спавн, Специалист) —
    один LLM-вызов на подтверждение, а не автоматическое одобрение."""

    MAX_PER_TASK_PER_ROLE = 3
    MAX_PARALLEL_PER_TASK = 6

    def __init__(self, gateway, cost_cap: float | None):
        self.gateway = gateway
        self.cost_cap = cost_cap

    def _current_cost(self) -> float:
        u = getattr(self.gateway, "usage", None)
        return u.actual_cost if u else 0.0

    def review(self, req: SpawnRequest, task: Task) -> AuditVerdict:
        if self.cost_cap is not None and self._current_cost() >= self.cost_cap:
            return AuditVerdict(0, f"бюджет исчерпан (${self._current_cost():.4f} >= ${self.cost_cap})")

        already = task.spawn_count.get(req.role, 0)
        if already >= self.MAX_PER_TASK_PER_ROLE and req.count > 0:
            return AuditVerdict(0, f"роль {req.role} уже спавнилась {already} раз на эту задачу — лимит")

        count = min(req.count, self.MAX_PARALLEL_PER_TASK)
        borderline = count > 3 or req.role == "specialist" or already > 0
        if not borderline:
            return AuditVerdict(count, "правила: одобрено без эскалации")

        # пограничный случай — спросить дешёвую модель, реально ли это надо, а не просто разрешить.
        # Раньше сюда не попадал task.feedback — ревизор не видел, что предыдущая попытка была
        # отклонена ревью, и дважды отказывал в повторном implementer после честного reject, приняв
        # его за произвольное раздувание роя. Причина повтора должна быть видна, не только счётчик.
        feedback_note = ""
        if task.feedback:
            feedback_note = "\n\nПрошлые попытки по этой задаче не приняты:\n" + "\n".join(
                f"- {f}" for f in task.feedback[-2:]
            )
        brief = (
            f"Диспетчер просит заспавнить {count} агентов роли \"{req.role}\" для задачи "
            f"\"{task.id}\" ({task.goal}). Причина: {req.reason}. Уже было спавнов этой роли на "
            f"эту задачу: {already}.{feedback_note}\n"
            "Это реально нужно, или диспетчер плодит агентов без повода? Если задача уже была "
            "отклонена (см. прошлые попытки выше) — новый спавн для исправления оправдан. В result "
            'верни JSON: {"approve_count": N, "why": "..."} — N не больше запрошенного, 0 если не обосновано.'
        )
        reply = call_agent_any(self.gateway, _with_fallback(AUDITOR_LLM_MODEL, AUDITOR_FALLBACK_MODEL),
                            "checker", f"audit-{task.id}-{req.role}", brief, max_tokens=2048)
        if reply.status != "done":
            # Живой прогон (shakedown-game, 2026-09-24): обе audit-модели легли (nemotron мёртвая
            # с прошлого спавна, gpt-5.6-luna тоже недоступна в моменте) -> каждый повторный спавн
            # после честного reject получал отказ 0 без единого реального суждения, застой
            # гарантирован за 3 итерации. Инфраструктурный отказ ("модель недоступна") — это не то
            # же самое, что содержательное решение "не обосновано". Раз модель физически не
            # ответила, падаем на тот же лимит, что и rule-based путь non-borderline случая (count,
            # уже урезанный лимитами MAX_PER_TASK_PER_ROLE/MAX_PARALLEL_PER_TASK выше) — не
            # неограниченно, но и не глухой запрет навсегда.
            return AuditVerdict(count, f"ревизор недоступен, одобрено по лимитам без LLM-подтверждения: "
                                        f"{reply.result or reply.raw.get('blocked_reason')}")
        try:
            verdict = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
            n = max(0, min(int(verdict.get("approve_count", 0)), count))
            return AuditVerdict(n, verdict.get("why", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            return AuditVerdict(0, "ревизор вернул нечитаемый вердикт — отказ по умолчанию")


# ---------- Диспетчер ----------

def dispatcher_plan(gateway, task: Task, board: Board, repo_context: str = "",
                     max_tokens: int = 2048) -> list[SpawnRequest]:
    """Решает состав для готовой задачи. Rule-based по kind + подтверждение размера волны у LLM
    только когда неочевидно (kind=generic). Вызывается один раз на задачу, до смены её статуса —
    что уже отработано (findings очищены), второй раз план не просят."""
    if task.kind == "research":
        return [SpawnRequest("researcher", 2, "разведка перед реализацией", task.id),
                SpawnRequest("analyst", 1, "риски и граничные случаи", task.id)]
    if task.kind == "implement":
        return [SpawnRequest("implementer", 1, "реализация задачи", task.id)]
    if task.kind == "generic":
        # неочевидный тип — спросить диспетчерскую модель, что вообще нужно
        brief = (
            f"Задача \"{task.id}\": {task.goal}. Реши, какие роли и сколько экземпляров нужны "
            "для её выполнения. Роли: researcher, analyst, implementer, specialist. В result верни "
            'JSON-массив: [{"role": "...", "count": N, "reason": "..."}]. Не спавнь больше, чем реально нужно.'
        )
        reply = call_agent_any(gateway, _with_fallback(DISPATCHER_MODEL), "dispatcher", task.id, brief,
                                repo_context, max_tokens=max_tokens)
        if reply.status == "done":
            try:
                raw = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
                return [SpawnRequest(r["role"], int(r.get("count", 1)), r.get("reason", ""), task.id) for r in raw]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return [SpawnRequest("specialist", 1, "диспетчер не распознал тип задачи, свободный агент", task.id)]
    return []


# Пул для ревью (раздел 6): минимум 3 РАЗНЫЕ модели, ни одна не автор проверяемого артефакта.
REVIEWER_POOL = ["gpt-6-sol", "gpt-6-luna", "gpt-5.6-luna", "claude-opus-5.5", FALLBACK_MODEL]


def dispatcher_reviewers_needed(task: Task, wave_replies: list[AgentReply]) -> list[str]:
    """Артефакт готов к ревью -> модели ревьюеров, исключая авторов. Пустой список — ревьюить нечего."""
    if not any(r.artifacts for r in wave_replies):
        return []
    authors = {r.model for r in wave_replies}
    pool = [m for m in REVIEWER_POOL if m not in authors]
    return pool[:3]


def review_quorum_verdict(gateway, task: Task, artifact_summary: str,
                           reviewer_models: list[str],
                           max_tokens: int = 4096) -> tuple[str, list[AgentReply], list[str]]:
    """Раздел 6 протокола: approve / approve_with_findings / reject. Любой reject — блокирует,
    расхождение не усредняется (голосов < 3 моделей просто не бывает большинства — решает наличие reject)."""
    # verdict/why описаны как поля верхнего уровня в contract.py::build_prompt (role="reviewer") —
    # здесь только предмет ревью, схему не повторяем: повторение двух версий инструкции (тут JSON-в-result,
    # там поле верхнего уровня) и путало живые модели, отсюда половина случаев "verdict отсутствует".
    brief = f"Ревью артефакта задачи \"{task.id}\" ({task.goal}). Что сдали:\n{artifact_summary}\n\nОцени: закрывает ли это задачу, нет ли явных ошибок."
    valid_verdicts = {"approve", "approve_with_findings", "reject"}

    def ask(model: str) -> AgentReply:
        return call_agent_any(gateway, [model, FALLBACK_MODEL] if model != FALLBACK_MODEL else [model],
                               "reviewer", f"review-{task.id}", brief, max_tokens=max_tokens)

    replies: list[AgentReply] = []
    valid_verdicts_list, reasons = [], []
    for m in reviewer_models:
        r = ask(m)
        v = r.raw.get("verdict")
        if v not in valid_verdicts:
            r = ask(m)  # одна попытка перезадать тому же ревьюеру, прежде чем считать голос потерянным
            v = r.raw.get("verdict")
        replies.append(r)
        if v in valid_verdicts:
            valid_verdicts_list.append(v)
            reasons.append(r.result or "")
        else:
            # раздел 6 говорит про reject-голос, а не про немую модель — это ГОЛОС ПОТЕРЯН (abstain), не reject.
            # Менеджер сам предложил эту схему после того, как один abstain душил approve от двух остальных.
            reasons.append(f"ревьюер ({r.model}) дважды не вернул понятный verdict, голос не учтён: {(r.result or '')[:200]}")

    if not valid_verdicts_list:
        final = "reject"  # ни одного читаемого голоса вообще — не молчаливое одобрение
    elif "reject" in valid_verdicts_list:
        final = "reject"
    elif "approve_with_findings" in valid_verdicts_list:
        final = "approve_with_findings"
    else:
        final = "approve"
    return final, replies, [r for r in reasons if r]


def dispatcher_checker_needed(task: Task, wave_replies: list[AgentReply]) -> SpawnRequest | None:
    """После волны: нужен ли Сверщик — на противоречия разведки, конфликт файлов реализации,
    или отдельное архитектурное решение одного из агентов."""
    if len(wave_replies) < 2:
        # реализатор в одиночку принял архитектурное решение (есть непустые assumptions) — тоже повод
        if wave_replies and wave_replies[0].raw.get("assumptions"):
            return SpawnRequest("checker", 1, "единственный реализатор сделал архитектурные допущения", task.id)
        return None
    results = [r.result for r in wave_replies]
    if len(set(results)) > 1:
        return SpawnRequest("checker", 1, "параллельные ответы разошлись — свести", task.id)
    paths = [a["path"] for r in wave_replies for a in r.artifacts]
    if len(paths) != len(set(paths)):
        return SpawnRequest("checker", 1, "параллельные задачи трогают одни файлы — конфликт", task.id)
    return None


# ---------- Приёмщик ----------

@dataclass
class AcceptanceResult:
    goal_met: bool
    ground_truth_ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.goal_met and self.ground_truth_ok


def acceptor_check_goal(gateway, original_goal: str, board: Board, max_tokens: int = 4096) -> tuple[bool, str]:
    # раньше сводка была одним словом ("reply"/"review") из последней записи history — приёмщик физически
    # не мог сверить содержимое с целью. Теперь реальные пути и содержимое файлов, как у ревьюеров.
    parts = [f"Исходная цель роя: {original_goal}\n"]
    for t in board.tasks.values():
        parts.append(f"- Задача \"{t.id}\" [{t.status}]: {t.goal}")
        for a in t.artifacts:
            if a.get("content"):
                # тот же кап, что у ревьюеров (orchestrator.py) — раньше 3000 символов резало крупные
                # файлы посреди функции, приёмщик честно не мог сверить обрезанный код с целью
                parts.append(f"  Файл {a.get('path')}:\n{a['content'][:60000]}")
    summary = "\n".join(parts)
    brief = (
        f"{summary}\n\n"
        "Сверь результат с исходной ЦЕЛЬЮ, не со списком задач — задачи могли быть выполнены "
        "формально и мимо цели."
    )
    reply = call_agent_any(gateway, _with_fallback(*ACCEPTOR_MODELS), "acceptor", "accept-goal", brief, max_tokens=max_tokens)
    if reply.status != "done":
        return False, reply.raw.get("blocked_reason", "приёмщик не смог оценить")
    v = reply.raw.get("goal_met")
    if not isinstance(v, bool):
        return False, f"приёмщик не вернул понятное goal_met ({v!r}): {(reply.result or '')[:300]}"
    return v, reply.result or ""


def acceptor_check_ground_truth(board: Board, allow_llm_verify_cmd: bool = False) -> tuple[bool, list[str], dict[str, str]]:
    """Не доверяет self_check агентов — реально выполняет verify_cmd каждой задачи на этой машине.
    Третий элемент — {task_id: причина отказа} для тех, кто провалил verify_cmd, чтобы вызывающий
    мог переоткрыть именно их с конкретной ошибкой, а не считать провал приёмки концом прогона.

    verify_cmd_source="llm" (вписан Менеджером при авто-декомпозиции цели, не человеком в YAML) НЕ
    исполняется, если allow_llm_verify_cmd не выставлен явно — иначе бесплатная модель без ревью
    получает произвольный shell-доступ к машине через subprocess.run(shell=True) ниже. Пропущенная
    команда падает не в failures, а трактуется как отсутствующий verify_cmd — приёмка остаётся на
    ревью-кворуме/self-report, как для любой задачи без verify_cmd вообще."""
    notes = []
    failures: dict[str, str] = {}
    ok = True
    for t in board.tasks.values():
        if not t.verify_cmd:
            continue
        if t.verify_cmd_source == "llm" and not allow_llm_verify_cmd:
            notes.append(
                f"{t.id}: verify_cmd от LLM пропущен без allow_llm_verify_cmd (не выполняется "
                f"на машине без ревью человека) — {t.verify_cmd}"
            )
            continue
        try:
            r = subprocess.run(t.verify_cmd, shell=True, capture_output=True, text=True, timeout=120)
            passed = r.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
            r = None
        note = f"{t.id}: verify_cmd {'ok' if passed else 'FAIL'} — {t.verify_cmd}"
        if not passed:
            fail_detail = f"код {r.returncode}, stderr: {r.stderr[:300]}" if r is not None else "таймаут"
            note += f" ({fail_detail})"
            failures[t.id] = fail_detail
        notes.append(note)
        ok = ok and passed
    return ok, notes, failures

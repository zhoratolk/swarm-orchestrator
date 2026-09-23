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

def manager_decompose(gateway, goal: str, max_tokens: int = 4096) -> list[Task]:
    """Раскладывает цель на задачи один раз в начале, если задачи не заданы в YAML руками."""
    brief = (
        f"Разложи цель на независимые (по возможности) задачи для роя. Цель:\n{goal}\n\n"
        "В result верни JSON-массив задач строкой: "
        '[{"id": "t1", "goal": "...", "deps": [], "kind": "research|implement|generic", "verify_cmd": null}, ...]. '
        "id короткие, deps — id других задач из этого же списка, kind подсказывает какого рода работа."
    )
    reply = call_agent_any(gateway, _with_fallback(MANAGER_MODEL), "manager", "decompose", brief, max_tokens=max_tokens)
    tasks: list[Task] = []
    if reply.status == "done":
        try:
            raw = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
            for t in raw:
                tasks.append(Task(id=t["id"], goal=t["goal"], deps=t.get("deps", []),
                                   kind=t.get("kind", "generic"), verify_cmd=t.get("verify_cmd")))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if not tasks:
        tasks = [Task(id="t1", goal=goal, deps=[], kind="generic")]
    return tasks


def manager_replan(gateway, board: Board, findings_summary: str, max_tokens: int = 2048) -> str:
    """Вызывается при застое/reject — Менеджер решает, что делать дальше. Возвращает свободный текст-решение."""
    brief = (
        f"Доска задач застряла. Находки: {findings_summary}\n"
        "Реши: какую задачу переоткрыть, что изменить в подходе, или что эскалировать человеку. "
        "Кратко в result."
    )
    reply = call_agent_any(gateway, _with_fallback(MANAGER_MODEL), "manager", "replan", brief, max_tokens=max_tokens)
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

        # пограничный случай — спросить дешёвую модель, реально ли это надо, а не просто разрешить
        brief = (
            f"Диспетчер просит заспавнить {count} агентов роли \"{req.role}\" для задачи "
            f"\"{task.id}\" ({task.goal}). Причина: {req.reason}. Уже было спавнов этой роли на "
            f"эту задачу: {already}.\n"
            "Это реально нужно, или диспетчер плодит агентов без повода? В result верни JSON: "
            '{"approve_count": N, "why": "..."} — N не больше запрошенного, 0 если не обосновано.'
        )
        reply = call_agent_any(self.gateway, _with_fallback(AUDITOR_LLM_MODEL), "checker", f"audit-{task.id}-{req.role}",
                            brief, max_tokens=512)
        if reply.status != "done":
            return AuditVerdict(0, f"ревизор не смог решить: {reply.result or reply.raw.get('blocked_reason')}")
        try:
            verdict = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
            n = max(0, min(int(verdict.get("approve_count", 0)), count))
            return AuditVerdict(n, verdict.get("why", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            return AuditVerdict(0, "ревизор вернул нечитаемый вердикт — отказ по умолчанию")


# ---------- Диспетчер ----------

def dispatcher_plan(gateway, task: Task, board: Board, max_tokens: int = 1024) -> list[SpawnRequest]:
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
        reply = call_agent_any(gateway, _with_fallback(DISPATCHER_MODEL), "dispatcher", task.id, brief, max_tokens=max_tokens)
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
                           reviewer_models: list[str], max_tokens: int = 1024) -> tuple[str, list[AgentReply]]:
    """Раздел 6 протокола: approve / approve_with_findings / reject. Любой reject — блокирует,
    расхождение не усредняется (голосов < 3 моделей просто не бывает большинства — решает наличие reject)."""
    brief = (
        f"Ревью артефакта задачи \"{task.id}\" ({task.goal}). Что сдали:\n{artifact_summary}\n\n"
        "Оцени: закрывает ли это задачу, нет ли явных ошибок. В result JSON: "
        '{"verdict": "approve|approve_with_findings|reject", "why": "..."}.'
    )
    replies = [
        call_agent_any(gateway, [m, FALLBACK_MODEL] if m != FALLBACK_MODEL else [m],
                        "reviewer", f"review-{task.id}", brief, max_tokens=max_tokens)
        for m in reviewer_models
    ]
    verdicts = []
    for r in replies:
        try:
            v = json.loads(r.result) if isinstance(r.result, str) else r.result
            verdicts.append(v.get("verdict", "reject"))
        except (json.JSONDecodeError, ValueError, TypeError):
            verdicts.append("reject")  # нечитаемый ответ ревьюера — не молчаливое одобрение
    if "reject" in verdicts:
        final = "reject"
    elif "approve_with_findings" in verdicts:
        final = "approve_with_findings"
    else:
        final = "approve"
    return final, replies


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


def acceptor_check_goal(gateway, original_goal: str, board: Board, max_tokens: int = 2048) -> tuple[bool, str]:
    summary = "\n".join(f"- {t.id} ({t.status}): {t.history[-1]['event'] if t.history else ''}"
                         for t in board.tasks.values())
    brief = (
        f"Исходная цель роя: {original_goal}\n\nИтог по задачам:\n{summary}\n\n"
        "Сверь результат с исходной ЦЕЛЬЮ, не со списком задач — задачи могли быть выполнены "
        "формально и мимо цели. В result JSON: {\"goal_met\": bool, \"why\": \"...\"}."
    )
    reply = call_agent_any(gateway, _with_fallback(*ACCEPTOR_MODELS), "acceptor", "accept-goal", brief, max_tokens=max_tokens)
    if reply.status != "done":
        return False, reply.raw.get("blocked_reason", "приёмщик не смог оценить")
    try:
        v = json.loads(reply.result) if isinstance(reply.result, str) else reply.result
        return bool(v.get("goal_met")), v.get("why", "")
    except (json.JSONDecodeError, ValueError, TypeError):
        return False, "приёмщик вернул нечитаемый вердикт"


def acceptor_check_ground_truth(board: Board) -> tuple[bool, list[str]]:
    """Не доверяет self_check агентов — реально выполняет verify_cmd каждой задачи на этой машине."""
    notes = []
    ok = True
    for t in board.tasks.values():
        if not t.verify_cmd:
            continue
        try:
            r = subprocess.run(t.verify_cmd, shell=True, capture_output=True, text=True, timeout=120)
            passed = r.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
            r = None
        note = f"{t.id}: verify_cmd {'ok' if passed else 'FAIL'} — {t.verify_cmd}"
        if not passed and r is not None:
            note += f" (код {r.returncode}, stderr: {r.stderr[:200]})"
        notes.append(note)
        ok = ok and passed
    return ok, notes

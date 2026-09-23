"""JSON-контракт агента (раздел 5 протокола): построение промпта, парсинг, валидация, повтор при браке."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .gateway import ModelUnavailable
from .scrub import scrub

REQUIRED_KEYS = {
    "role", "task_id", "status", "confidence", "result",
    "artifacts", "tools_used", "findings", "assumptions", "self_check",
}
VALID_STATUS = {"done", "blocked", "needs_info"}
VALID_ROLES = {
    "manager", "dispatcher", "researcher", "analyst", "implementer",
    "checker", "reviewer", "specialist", "acceptor",
}


@dataclass
class AgentReply:
    raw: dict
    model: str
    role_asked: str

    @property
    def status(self) -> str:
        return self.raw.get("status", "blocked")

    @property
    def confidence(self) -> float:
        return float(self.raw.get("confidence") or 0.0)

    @property
    def result(self) -> str:
        return self.raw.get("result", "")

    @property
    def findings(self) -> list[dict]:
        return self.raw.get("findings", []) or []

    @property
    def blockers(self) -> list[dict]:
        return [f for f in self.findings if f.get("severity") == "blocker"]

    @property
    def artifacts(self) -> list[dict]:
        return self.raw.get("artifacts", []) or []


class ContractViolation(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    # модели любят обернуть JSON в ```json ... ``` — снимаем ограждение
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # либо просто найти первую { и последнюю } — на случай преамбулы/постскриптума
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _validate(data: dict) -> list[str]:
    problems = []
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        problems.append(f"нет полей: {sorted(missing)}")
    if data.get("status") not in VALID_STATUS:
        problems.append(f"status должен быть одним из {VALID_STATUS}, получено {data.get('status')!r}")
    if data.get("status") == "done" and not str(data.get("result", "")).strip():
        problems.append("status=done, но result пустой")
    if data.get("status") == "blocked" and not str(data.get("blocked_reason", "")).strip():
        problems.append("status=blocked, но blocked_reason пустой")
    conf = data.get("confidence")
    if conf is not None and not (0.0 <= float(conf) <= 1.0):
        problems.append(f"confidence вне [0,1]: {conf}")
    blockers = [f for f in (data.get("findings") or []) if f.get("severity") == "blocker"]
    if blockers and float(conf or 0) > 0.8:
        problems.append("confidence > 0.8 при непустых блокерах — противоречие, объяснить")
    return problems


def build_prompt(role: str, task_id: str, brief: str, extra_context: str = "") -> str:
    scrubbed, found = scrub(brief + "\n" + extra_context)
    secret_note = f"\n(из текста задачи вычищено: {', '.join(sorted(set(found)))} — не восстанавливать)" if found else ""
    schema = (
        '{"role": "...", "task_id": "...", "spawned_by": "...", "status": "done|blocked|needs_info", '
        '"confidence": 0.0, "result": "...", "artifacts": [...], "tools_used": [...], "findings": [...], '
        '"assumptions": [...], "blocked_reason": "...", "self_check": {"goal_met": false, "why": "..."}}'
    )
    return (
        f"Ты — агент роя в роли {role}. task_id: {task_id}.\n\n"
        f"Задача:\n{scrubbed}{secret_note}\n\n"
        f"Верни ТОЛЬКО валидный JSON, без пояснений вокруг, строго по схеме:\n{schema}\n"
        f"role в ответе обязан быть \"{role}\". Если чего-то не хватает для выполнения — status=\"needs_info\", "
        f"опиши в result что именно нужно. Если задача невыполнима — status=\"blocked\" и заполни blocked_reason."
    )


def call_agent(gateway, model: str, role: str, task_id: str, brief: str, extra_context: str = "",
                max_tokens: int = 4096, max_repairs: int = 2) -> AgentReply:
    """Зовёт агента, парсит и валидирует JSON-контракт, при браке — до max_repairs повторов с объяснением ошибки."""
    prompt = build_prompt(role, task_id, brief, extra_context)
    last_error = ""
    for attempt in range(max_repairs + 1):
        full_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nПредыдущий ответ не прошёл проверку: {last_error}\n"
            f"Пришли заново — только исправленный JSON, без текста вокруг."
        )
        result = gateway.call(model, full_prompt, max_tokens=max_tokens)
        try:
            data = _extract_json(result.text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"не удалось распарсить JSON: {e}"
            continue
        problems = _validate(data)
        if problems:
            last_error = "; ".join(problems)
            continue
        data.setdefault("role", role)
        return AgentReply(raw=data, model=model, role_asked=role)

    # исчерпали повторы — синтетический blocked-ответ, чтобы цикл не падал, а видел проблему
    return AgentReply(
        raw={
            "role": role, "task_id": task_id, "spawned_by": "contract-repair-exhausted",
            "status": "blocked", "confidence": 0.0,
            "result": "", "artifacts": [], "tools_used": [], "findings": [],
            "assumptions": [], "blocked_reason": f"агент не смог вернуть валидный контракт: {last_error}",
            "self_check": {"goal_met": False, "why": "contract violation"},
        },
        model=model, role_asked=role,
    )

"""Доска задач: состояние, зависимости, персистентность (--resume), история под каждой задачей."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATUSES = ("queued", "dispatched", "in_progress", "needs_review", "blocked", "done")


@dataclass
class Task:
    id: str
    goal: str
    deps: list[str] = field(default_factory=list)
    verify_cmd: str | None = None
    kind: str = "generic"  # research | implement | generic — подсказка Диспетчеру
    status: str = "queued"
    history: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    spawn_count: dict = field(default_factory=dict)  # role -> сколько раз спавнили, для Ревизора
    feedback: list[str] = field(default_factory=list)  # причины прошлых reject/blocked — копится, не теряется при переоткрытии

    def log(self, event: str, **kw):
        self.history.append({"t": time.time(), "event": event, **kw})

    def brief(self) -> str:
        """Задача для агента: цель плюс то, что не получилось в прошлые заходы, если было."""
        if not self.feedback:
            return self.goal
        past = "\n".join(f"- {f}" for f in self.feedback[-3:])
        return (f"{self.goal}\n\nПрошлые попытки не приняты, учти это в этой:\n{past}")


class Board:
    def __init__(self, tasks: dict[str, Task], path: Path):
        self.tasks = tasks
        self.path = path

    @classmethod
    def new(cls, tasks: list[Task], path: Path) -> "Board":
        return cls({t.id: t for t in tasks}, path)

    @classmethod
    def load(cls, path: Path) -> "Board":
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = {tid: Task(**t) for tid, t in data.items()}
        return cls(tasks, path)

    def save(self):
        data = {tid: asdict(t) for tid, t in self.tasks.items()}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def ready(self) -> list[Task]:
        """Задачи в queued, у которых все зависимости done."""
        out = []
        for t in self.tasks.values():
            if t.status != "queued":
                continue
            if all(self.tasks.get(d) and self.tasks[d].status == "done" for d in t.deps):
                out.append(t)
        return out

    def pending(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status not in ("done",)]

    def blockers(self) -> list[dict]:
        out = []
        for t in self.tasks.values():
            out += t.findings
        return [f for f in out if f.get("severity") == "blocker"]

    def all_done(self) -> bool:
        return all(t.status == "done" for t in self.tasks.values())

    def any_permanently_blocked(self) -> bool:
        return any(t.status == "blocked" for t in self.tasks.values())

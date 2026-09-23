"""Грубый regex-фильтр секретов и персональных данных перед отправкой наружу (раздел 9).

Не панацея — для по-настоящему чувствительных данных шлюз вообще не использовать.
"""
from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(sk|xpl|ghp|gho|ghs|AKIA)[-_A-Za-z0-9]{10,}\b"), "API_KEY"),
    (re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"), "TOKEN_LIKE"),
    (re.compile(r"(?i)\b(password|пароль|passwd)\s*[:=]\s*\S+"), "PASSWORD"),
    (re.compile(r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+"), "DB_CONNECTION"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "EMAIL"),
    (re.compile(r"\+?\d[\d\s()\-]{8,}\d"), "PHONE"),
    (re.compile(r"\b\d{12}\b"), "IIN_BIN_LIKE"),
    (re.compile(r"(?i)\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "CARD_LIKE"),
]


def scrub(text: str) -> tuple[str, list[str]]:
    """Возвращает (очищенный текст, список типов найденных секретов) — типы, не значения."""
    found: list[str] = []
    out = text
    for pattern, tag in _PATTERNS:
        def repl(m, tag=tag):
            found.append(tag)
            return f"<SECRET:{tag}>"
        out = pattern.sub(repl, out)
    return out, found


def scrub_env_files(text: str) -> str:
    """.env-подобные блоки (KEY=value построчно) — на всякий случай отдельно, паттерны выше их частично ловят."""
    lines = text.splitlines()
    out = []
    for line in lines:
        if re.match(r"^\s*[A-Z_][A-Z0-9_]*\s*=\s*\S+", line) and any(
            k in line.upper() for k in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD")
        ):
            name = line.split("=", 1)[0].strip()
            out.append(f"{name}=<SECRET:ENV_VALUE>")
        else:
            out.append(line)
    return "\n".join(out)

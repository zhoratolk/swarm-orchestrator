"""HTTP-клиент шлюза Experiential Labs: модели, ретраи, учёт стоимости.

Ключ только из окружения (EXPLABS_API_KEY). --dry-run подменяет call() на FakeGateway,
чтобы весь цикл можно было проверить без ключа и без сети.
"""
from __future__ import annotations

import os
import time
import json
import random
from dataclasses import dataclass, field

import requests

BASE = os.environ.get("EXPLABS_BASE", "https://api.experientiallabs.ai")
API_VERSION = "2023-06-01"

# Раздел 3 протокола. no_training важен для scrub.py и roles.py (кому нельзя чужие данные).
MODELS = {
    "claude-opus-5.5": {"free": True, "no_training": True, "gpt_family": False},
    "gpt-6-sol": {"free": True, "no_training": True, "gpt_family": True},
    "gpt-6-luna": {"free": True, "no_training": True, "gpt_family": True},
    "gpt-5.6-luna": {"free": True, "no_training": False, "gpt_family": True},
    "nemotron-3-ultra-550b-a55b": {"free": True, "no_training": False, "gpt_family": False},
    "gpt-6-astra": {"free": False, "no_training": False, "gpt_family": True},
    "claude-sonnet-5": {"free": False, "no_training": True, "gpt_family": False},
    "gpt-5.6-terra": {"free": False, "no_training": False, "gpt_family": True},
    "qwen3.8-27b": {"free": False, "no_training": False, "gpt_family": False},
    "deepseek-v4-flash": {"free": False, "no_training": False, "gpt_family": False},
}

UNAVAILABLE = {
    "claude-fable-5.1": "заблокирована тарифом, 429 model_requires_purchase",
    "jev": "не Messages-бэкенд, отдельный /v1/systemone, агентом роя не использовать",
}

# Раздел 10 протокола: прикол вручную по openrouter.ai/api/v1/models, не живой фетч.
# Ориентир $/1M токенов (вход, выход), грубо по классу модели. Сверить руками при дрейфе цен.
SHADOW_PRICES = {
    "claude-opus-5.5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "gpt-6-sol": (2.5, 10.0),
    "gpt-6-luna": (0.6, 2.4),
    "gpt-6-astra": (2.5, 10.0),
    "gpt-5.6-luna": (0.6, 2.4),
    "gpt-5.6-terra": (0.15, 0.6),
    "nemotron-3-ultra-550b-a55b": (0.4, 1.6),
    "qwen3.8-27b": (0.2, 0.8),
    "deepseek-v4-flash": (0.1, 0.3),
}


class GatewayError(RuntimeError):
    pass


class ModelUnavailable(GatewayError):
    """429 model_requires_purchase — модель исключается из пула на весь прогон."""


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    is_byok: bool = False

    def shadow_cost(self, model: str) -> float:
        lo, hi = SHADOW_PRICES.get(model, (1.0, 3.0))
        return self.input_tokens / 1_000_000 * lo + self.output_tokens / 1_000_000 * hi


class Usage:
    """Копится за весь прогон: два счётчика, факт и тень (раздел 10)."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.actual_cost = 0.0
        self.shadow_cost = 0.0
        self.by_model: dict[str, dict] = {}

    def add(self, model: str, result: CallResult):
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.actual_cost += result.cost
        sc = result.shadow_cost(model)
        self.shadow_cost += sc
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "shadow": 0.0})
        m["calls"] += 1
        m["in"] += result.input_tokens
        m["out"] += result.output_tokens
        m["cost"] += result.cost
        m["shadow"] += sc

    def report(self) -> str:
        lines = [
            "| Показатель | Значение |", "|---|---|",
            f"| Вызовов всего | {self.calls} |",
            f"| Токенов вход / выход | {self.input_tokens} / {self.output_tokens} |",
            f"| Фактически потрачено | ${self.actual_cost:.4f} |",
            f"| Стоило бы по прайсу | ${self.shadow_cost:.4f} |",
            "", "| Модель | Вызовов | Вход | Выход | Факт | Тень |", "|---|---|---|---|---|---|",
        ]
        for model, m in sorted(self.by_model.items()):
            lines.append(f"| {model} | {m['calls']} | {m['in']} | {m['out']} | ${m['cost']:.4f} | ${m['shadow']:.4f} |")
        return "\n".join(lines)


class Gateway:
    """Реальный шлюз. call() ретраит 429/5xx с экспоненциальной задержкой (раздел 11)."""

    def __init__(self, usage: Usage | None = None, timeout: float = 90.0):
        self.key = os.environ.get("EXPLABS_API_KEY")
        if not self.key:
            raise GatewayError(
                "EXPLABS_API_KEY не задан. export EXPLABS_API_KEY=xpl_... или .env (см. env.example). "
                "Для проверки логики без ключа: --dry-run."
            )
        self.usage = usage or Usage()
        self.timeout = timeout
        self.dead_models: set[str] = set()

    def call(self, model: str, prompt: str, max_tokens: int = 4096) -> CallResult:
        if model in UNAVAILABLE:
            raise ModelUnavailable(f"{model}: {UNAVAILABLE[model]}")
        if model in self.dead_models:
            raise ModelUnavailable(f"{model}: устойчивый 429 в этом прогоне, выведена из пула")
        info = MODELS.get(model, {})
        if info.get("gpt_family") and max_tokens < 16:
            max_tokens = 16  # раздел 3: GPT-семейство требует max_tokens >= 16

        headers = {
            "Authorization": f"Bearer {self.key}",
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}

        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = requests.post(f"{BASE}/v1/messages", headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(delay + random.uniform(0, 0.3))
                delay *= 2
                continue

            if r.status_code == 429:
                try:
                    err = r.json().get("error", {})
                except ValueError:
                    err = {}
                if err.get("code") == "model_requires_purchase" or err.get("type") == "model_requires_purchase":
                    raise ModelUnavailable(f"{model}: тариф не позволяет, исключена из пула")
                last_err = GatewayError(f"429 от {model} (попытка {attempt + 1})")
                time.sleep(delay + random.uniform(0, 0.3))
                delay *= 2
                continue

            if r.status_code == 403:
                try:
                    err = r.json().get("error", {})
                except ValueError:
                    err = {}
                msg = err.get("message", r.text[:200])
                # постоянный отказ (обычно геоблок апстрима у Anthropic/OpenAI-семейств), не транзиент —
                # ретраить бессмысленно, сразу помечаем мёртвой на этот прогон и уходим на фолбэк выше по стеку
                self.dead_models.add(model)
                raise ModelUnavailable(f"{model}: 403 {msg}")

            if r.status_code >= 500:
                last_err = GatewayError(f"{r.status_code} от {model} (попытка {attempt + 1})")
                time.sleep(delay + random.uniform(0, 0.3))
                delay *= 2
                continue

            if r.status_code == 400:
                try:
                    msg = r.json().get("error", {}).get("message", r.text)
                except ValueError:
                    msg = r.text
                if "max_tokens" in msg and "16" in msg:
                    max_tokens = max(max_tokens, 16)
                    body["max_tokens"] = max_tokens
                    continue
                raise GatewayError(f"400 от {model}: {msg}")

            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
            result = CallResult(
                text=text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cost=usage.get("cost", 0.0),
                is_byok=usage.get("is_byok", False),
            )
            self.usage.add(model, result)
            return result

        # 5 попыток исчерпаны на 429 — устойчиво лежит, выводим из пула на этот прогон (раздел 11)
        self.dead_models.add(model)
        raise GatewayError(f"{model} не ответила за 5 попыток: {last_err}")


class FakeGateway:
    """--dry-run: детерминированная заглушка, без ключа и без сети. Эхо валидного JSON-контракта."""

    def __init__(self, usage: Usage | None = None):
        self.usage = usage or Usage()
        self.dead_models: set[str] = set()
        self._n = 0

    def call(self, model: str, prompt: str, max_tokens: int = 4096) -> CallResult:
        self._n += 1
        role = "specialist"
        for candidate in ("manager", "dispatcher", "researcher", "analyst", "implementer",
                           "checker", "reviewer", "acceptor"):
            if f'"role": "{candidate}' in prompt or f"role={candidate}" in prompt or candidate in prompt.lower()[:400]:
                role = candidate
                break
        payload = {
            "role": role, "task_id": f"dry-{self._n}", "spawned_by": "dry-run",
            "status": "done", "confidence": 0.7,
            "result": f"[dry-run] заглушка ответа {role} #{self._n} на модели {model}",
            "artifacts": [], "tools_used": [], "findings": [], "assumptions": ["dry-run: реальный вызов не делался"],
            "blocked_reason": "", "self_check": {"goal_met": True, "why": "dry-run"},
        }
        text = json.dumps(payload, ensure_ascii=False)
        result = CallResult(text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4, cost=0.0)
        self.usage.add(model, result)
        return result

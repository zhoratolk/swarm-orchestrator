---
name: swarm-orchestrator
description: Use when the user wants to run an autonomous multi-model swarm on a task via the Experiential Labs gateway (free Claude/GPT/Nemotron models over HTTP, not Claude Code subagents) — decomposing a goal into tasks, dispatching researchers/analysts/implementers/reviewers adaptively, and verifying the real end state before declaring success. Trigger phrases: "запусти рой", "swarm-orchestrator", "прогони через рой", "автономный рой на шлюзе".
---

# swarm-orchestrator

Код и документы живут в корне репозитория, не в этой папке — здесь только точка входа для Claude Code, чтобы скилл подхватывался автоматически. Смотреть:

- [`../../../README.md`](../../../README.md) — как запускать
- [`../../../PROTOCOL.md`](../../../PROTOCOL.md) — полная спека ролей
- [`../../../swarm/`](../../../swarm) — реализация
- [`../../../examples/example_task.yaml`](../../../examples/example_task.yaml) — пример задачи

## Когда использовать

Юзер просит прогнать задачу автономно, «запустил и забыл»: разложить цель на подзадачи, раздать нескольким моделям параллельно, свести, отревьюить, сдать. Не для быстрых одноразовых правок — для задач, которые реально выигрывают от параллельного роя разных моделей и от проверки результата независимыми проверяющими.

## Быстрый старт (из корня репозитория)

```bash
pip install -r requirements.txt
cp env.example .env   # вписать EXPLABS_API_KEY, ключ никогда не в промпт/код/лог
python -m swarm.orchestrator run examples/example_task.yaml --dry-run   # без ключа, проверить что цикл ходит
python -m swarm.orchestrator run examples/example_task.yaml             # с реальным ключом
```

Подробности запуска, устройства ролей и известные грабли — в корневом README и PROTOCOL.md.

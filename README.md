# swarm-orchestrator

Автономный рой моделей поверх шлюза [Experiential Labs](https://api.experientiallabs.ai) (Claude/GPT/Nemotron по HTTP, бесплатно). Не субагенты Claude Code — обычный Python-скрипт, который сам стучится к шлюзу параллельно, раздаёт задачи, проверяет результат.

Это одновременно:

1. **Обычный Python-инструмент** — код в [`swarm/`](swarm), запускается из корня репозитория, работает без Claude Code вообще.
2. **Claude Code skill** — [`.claude/skills/swarm-orchestrator/SKILL.md`](.claude/skills/swarm-orchestrator/SKILL.md) подхватывается автоматически, если открыть этот репозиторий (или скопировать `.claude/skills/swarm-orchestrator` вместе с корневым кодом в свой проект) под Claude Code.

## Быстрый старт

```bash
pip install -r requirements.txt
cp env.example .env   # вписать EXPLABS_API_KEY (никогда не коммитить .env, он в .gitignore)

python -m swarm.orchestrator run examples/example_task.yaml --dry-run   # без ключа, проверить что цикл ходит
python -m swarm.orchestrator run examples/example_task.yaml             # с реальным ключом
```

`run.log` в рабочей директории (по умолчанию `run/`, флаг `--run-dir`) — подробный лог. `board.json` — доска задач и чекпоинт, `--resume` продолжает прерванный прогон. `report.md` в конце — сколько вызовов, токенов, факт/теневая стоимость, что приняли и почему.

Полная спека ролей — [`PROTOCOL.md`](PROTOCOL.md).

## Запуск скилла в других чатах и проектах

Просто открыть этот репозиторий в Claude Code — скилл подхватится сам (`.claude/skills/swarm-orchestrator/SKILL.md` уже в репо). Для остальных случаев:

**Без скилла, из любого чата.** Скилл не обязателен — Claude Code (и просто ты сам) может запускать инструмент напрямую, если знает путь:

```bash
cd /путь/до/swarm-orchestrator && python -m swarm.orchestrator run examples/example_task.yaml
```

Скажи Claude в любой сессии: «есть swarm-orchestrator в `<путь>`, прогони на нём вот такую задачу» — этого достаточно, скилл только автоматизирует распознавание.

**Глобальный скилл — работает во всех проектах и чатах на этой машине без копирования.** Клонировать репозиторий прямо в папку скиллов пользователя и положить туда же корневой `SKILL.md` (внутри репо он на три уровня глубже, с относительными путями `../../../`; для глобального расположения нужен вариант с путями от корня самого себя):

```bash
# bash / git bash
git clone https://github.com/zhoratolk/swarm-orchestrator ~/.claude/skills/swarm-orchestrator
cp ~/.claude/skills/swarm-orchestrator/.claude/skills/swarm-orchestrator/SKILL.md ~/.claude/skills/swarm-orchestrator/SKILL.md
sed -i 's#\.\./\.\./\.\./##g' ~/.claude/skills/swarm-orchestrator/SKILL.md   # ../../../README.md -> README.md
cd ~/.claude/skills/swarm-orchestrator && pip install -r requirements.txt && cp env.example .env   # вписать ключ самому
```

```powershell
# PowerShell (Windows)
git clone https://github.com/zhoratolk/swarm-orchestrator "$env:USERPROFILE\.claude\skills\swarm-orchestrator"
$skill = "$env:USERPROFILE\.claude\skills\swarm-orchestrator"
Copy-Item "$skill\.claude\skills\swarm-orchestrator\SKILL.md" "$skill\SKILL.md"
(Get-Content "$skill\SKILL.md") -replace '\.\./\.\./\.\./', '' | Set-Content "$skill\SKILL.md"
Set-Location $skill; pip install -r requirements.txt; Copy-Item env.example .env   # вписать ключ самому
```

После этого в любом проекте на машине фраза «запусти рой» и подобные (раздел `description` в `SKILL.md`) сами подтягивают скилл — копировать код в каждый проект не нужно.

## Устройство вкратце

- **Менеджер** раскладывает цель на задачи, держит доску состояний, перепланирует при застое.
- **Диспетчер** — единственный, кто спавнит агентов; решает состав под каждую задачу.
- **Ревизор спавна** — следит за Диспетчером: не даёт плодить агентов без нужды (жёсткие лимиты + LLM-подтверждение на пограничные случаи).
- **Ревьюеры** (кворум ≥3 разные модели, не автор) — на каждый готовый артефакт, любой `reject` блокирует.
- **Приёмщик** не верит агентам на слово: сверяет результат с исходной целью и сам выполняет `verify_cmd` — реальную команду на машине, а не доверяет `self_check`.

Всё это должно работать «запустил и ушёл»: цикл сам крутится до успеха, доказанной невозможности, застоя (3 итерации без прогресса) или потолка бюджета — см. раздел 8 `PROTOCOL.md`.

## Известное из живых прогонов

Прогонялось по-настоящему (не только `--dry-run`) на реальном ключе. Что вылезло и как учтено — см. `git log`: часть бесплатных моделей (Anthropic/OpenAI-семейства) отдаёт 403 по геолокации с некоторых сетей, шлюз тогда автоматически откатывается на модели, которые отвечают; модели не всегда держат JSON-схему буквально (например кладут `findings` строками вместо объектов) — контракт нормализует форму вместо того, чтобы просто ронять ответ.

## Файлы

- `PROTOCOL.md` — полная спека ролей и правил, читать первым, если меняешь логику.
- `swarm/gateway.py` — HTTP-клиент шлюза: ретраи, ограничения моделей, фолбэк, прайс-таблица.
- `swarm/contract.py` — JSON-контракт агента: построение промпта, парсинг, нормализация, валидация, повтор при браке.
- `swarm/board.py` — доска задач, персистентность, резюме прогона.
- `swarm/scrub.py` — фильтр секретов/ПД перед отправкой наружу.
- `swarm/roles.py` — Менеджер, Диспетчер, Ревизор, Ревью-кворум, Приёмщик: построение промптов и решения.
- `swarm/orchestrator.py` — главный цикл и CLI (`run`, `--resume`, `--dry-run`).

## Лицензия

MIT, см. [`LICENSE`](LICENSE).

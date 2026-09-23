"""Regex-фильтр секретов/ПД перед отправкой наружу."""
from __future__ import annotations

from swarm.scrub import scrub, scrub_env_files


def test_scrub_masks_api_key_and_reports_type():
    out, found = scrub("ключ: sk-abcdefghij1234567890")
    assert "sk-abcdefghij1234567890" not in out
    assert "<SECRET:API_KEY>" in out
    assert found == ["API_KEY"]


def test_scrub_masks_email():
    out, found = scrub("пиши на ivan.petrov@example.com пожалуйста")
    assert "ivan.petrov@example.com" not in out
    assert "EMAIL" in found


def test_scrub_masks_password_assignment():
    out, found = scrub("password: hunter2 в конфиге")
    assert "hunter2" not in out
    assert "PASSWORD" in found


def test_scrub_masks_db_connection_string():
    out, found = scrub("строка подключения postgres://user:pass@host:5432/db")
    assert "postgres://user:pass@host:5432/db" not in out
    assert "DB_CONNECTION" in found


def test_scrub_no_secrets_returns_text_unchanged_and_empty_list():
    text = "обычный текст задачи без секретов"
    out, found = scrub(text)
    assert out == text
    assert found == []


def test_scrub_multiple_secret_types_all_reported():
    out, found = scrub("email a@b.com и password: qwerty123456")
    assert set(found) >= {"EMAIL", "PASSWORD"}


def test_scrub_env_files_masks_key_lines_keeps_others():
    text = "APP_NAME=demo\nAPI_KEY=xpl_abcdef123456\nPORT=8080"
    out = scrub_env_files(text)
    lines = out.splitlines()
    assert "APP_NAME=demo" in lines
    assert "PORT=8080" in lines
    assert "API_KEY=<SECRET:ENV_VALUE>" in lines
    assert "xpl_abcdef123456" not in out

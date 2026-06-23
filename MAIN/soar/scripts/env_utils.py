#!/usr/bin/env python3
"""Minimal helpers for loading local .env values without external deps."""

from __future__ import annotations

import os
from pathlib import Path


def find_env_file(start: Path | None = None, env_name: str = ".env") -> Path | None:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / env_name
        if candidate.is_file():
            return candidate
    return None


def load_dotenv_values(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_single_secret(env_file: Path) -> str:
    lines = [l.strip() for l in env_file.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) == 1 and "=" not in lines[0]:
        return lines[0]
    return ""


def resolve_env_value(name: str, current_value: str = "", start: Path | None = None) -> str:
    if current_value:
        return current_value
    existing = os.environ.get(name, "")
    if existing:
        return existing
    env_file = find_env_file(start=start)
    if not env_file:
        return ""
    values = load_dotenv_values(env_file)
    if name in values:
        return values[name]
    if name.endswith("_API_KEY"):
        return load_single_secret(env_file)
    return ""


def resolve_claude_key(current_value: str = "", start: Path | None = None) -> str:
    """Resolve LLM API key: checks argument value, then claude_API_KEY env, then .env."""
    if current_value:
        return current_value
    existing = os.environ.get("claude_API_KEY", "")
    if existing:
        return existing
    result = resolve_env_value("claude_API_KEY", current_value, start)
    if result:
        return result
    return ""

# Alias for compatibility
resolve_dashscope_key = resolve_claude_key

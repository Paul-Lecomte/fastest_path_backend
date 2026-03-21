# This module provides configuration utilities for the application, including environment variable handling and logging setup.
import os
import logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def setup_logging() -> None:
    if load_dotenv is not None:
        load_dotenv()
    level_name = get_env("LOG_LEVEL", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_neo4j_config() -> dict:
    return {
        "uri": get_env("NEO4J_URI", "bolt://localhost:7687"),
        "user": get_env("NEO4J_USER", "neo4j"),
        "password": get_env("NEO4J_PASSWORD", "neo4j"),
    }


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_network_cache_config() -> dict:
    max_age_seconds = _parse_int(get_env("NETWORK_CACHE_MAX_AGE_SECONDS", "0"), 0)
    return {
        "enabled": _parse_bool(get_env("NETWORK_CACHE_ENABLED", "true"), True),
        "path": get_env("NETWORK_CACHE_PATH", ".cache/transit_network.pkl"),
        "force_refresh": _parse_bool(get_env("NETWORK_CACHE_FORCE_REFRESH", "false"), False),
        "max_age_seconds": max_age_seconds if max_age_seconds > 0 else None,
    }

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

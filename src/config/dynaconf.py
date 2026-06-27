"""Dynaconf settings loader."""

from pathlib import Path

from dynaconf import Dynaconf  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]

dynaconf_settings = Dynaconf(
    root_path=ROOT,
    envvar_prefix="APP",
    settings_files=["config/settings.yaml"],
    secrets="config/.secrets.yaml",
    environments=True,
    load_dotenv=True,
    env_switcher="ENV_FOR_DYNACONF",
    merge_enabled=True,
)

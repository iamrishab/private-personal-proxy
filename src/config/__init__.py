"""Application configuration via Dynaconf."""

from config.dynaconf import dynaconf_settings
from config.models import Settings, get_settings

__all__ = ["Settings", "dynaconf_settings", "get_settings"]

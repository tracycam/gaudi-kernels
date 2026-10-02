"""Explicit engine configuration and dispatch; importing this acquires no device."""
from .config import EngineConfig, ConfigError, load_config
from .dispatch import DispatchTable, Request

__all__ = ['EngineConfig', 'ConfigError', 'load_config', 'DispatchTable', 'Request']

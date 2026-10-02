"""Explicit, metadata-only static-row dispatch. No environment or device reads."""
from gaudi_kernels.engine.config import ConfigError
_ROWS = (1,)
_COUNTS = {}

def validate_rows(rows):
    if type(rows) is not tuple or rows not in ((), (1,), (1, 8), tuple(range(1, 9))):
        raise ConfigError('Unqualified compact row set')

def set_rows(rows):
    global _ROWS
    validate_rows(rows)
    _ROWS = rows
    _COUNTS.clear()
    return snapshot()

def select_mode(mode, rows):
    if type(rows) is not int or rows < 0:
        raise ConfigError('Static token rows must be a nonnegative integer')
    if mode not in ('auto', 'compact', 'broadcast', 'sorted'):
        raise ConfigError('Unknown MoE mode')
    chosen = ('compact' if rows in _ROWS else 'broadcast') if mode == 'auto' else mode
    counters = _COUNTS.setdefault(str(rows), {})
    counters[chosen] = counters.get(chosen, 0) + 1
    return chosen

def snapshot():
    return {'compact_rows': list(_ROWS), 'mode': 'auto', 'python_capture_calls_by_rows': {k: dict(v) for (k, v) in _COUNTS.items()}, 'scope': 'static metadata at capture; not device invocation counts'}

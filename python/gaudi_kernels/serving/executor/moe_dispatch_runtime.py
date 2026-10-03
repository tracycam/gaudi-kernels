"""Explicit, metadata-only static-row dispatch. No environment or device reads."""
from gaudi_kernels.engine.config import ConfigError
from contextlib import contextmanager
from contextvars import ContextVar
_ROWS = (1,)
_COUNTS = {}
_DIAGNOSTIC_ROWS = ContextVar('private_compact_moe_rows', default=())

@contextmanager
def diagnostic_compact_rows(rows):
    """Scoped private-cycle probe, not serving admission or a config override.

    Unchanged-ELF component tests passed12 rows.16/24 did not show a complete
    chain win and are deliberately not admitted by this experiment.
    """
    if rows != (12,) or type(rows) is not tuple or _DIAGNOSTIC_ROWS.get():
        raise ConfigError('Only a nonnested private12-row experiment is admitted')
    token = _DIAGNOSTIC_ROWS.set(rows)
    try:
        yield
    finally:
        _DIAGNOSTIC_ROWS.reset(token)

def scale_tail_row_allowed(rows):
    return rows <= 8 or rows in _DIAGNOSTIC_ROWS.get()

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
    chosen = ('compact' if rows in _ROWS or rows in _DIAGNOSTIC_ROWS.get() else 'broadcast') if mode == 'auto' else mode
    counters = _COUNTS.setdefault(str(rows), {})
    counters[chosen] = counters.get(chosen, 0) + 1
    return chosen

def snapshot():
    return {'compact_rows': list(_ROWS), 'mode': 'auto', 'python_capture_calls_by_rows': {k: dict(v) for (k, v) in _COUNTS.items()}, 'scope': 'static metadata at capture; not device invocation counts'}

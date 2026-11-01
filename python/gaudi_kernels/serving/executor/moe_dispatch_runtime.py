"""Explicit, metadata-only static-row dispatch. No environment or device reads."""
from gaudi_kernels.engine.config import ConfigError, MoEDispatch
from contextlib import contextmanager
from contextvars import ContextVar
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

class DispatchState:
    """One model owns its immutable selection and capture counters."""
    def __init__(self, policy, *, top_k=8, num_experts=384):
        if type(policy) is not MoEDispatch:
            raise ConfigError('Typed MoE dispatch policy required')
        if (type(policy.grouped_min_mme_rows) is not int or
                not 1 <= policy.grouped_min_mme_rows <= 4096 or
                type(policy.grouped_slot_budget) is not int or
                not 0 <= policy.grouped_slot_budget <= 1536 or
                type(policy.grouped_padded_row_budget) is not int or
                not 0 <= policy.grouped_padded_row_budget <= 131072):
            raise ConfigError('Invalid grouped expert threshold or total slot budget')
        validate_rows(policy.compact_rows)
        self.policy = policy
        self.grouped_rows = policy.grouped_rows
        self.top_k = top_k
        self.num_experts = num_experts
        self.counts = {}
        self.plans = {}
        self._plans = self.plans
        if policy.grouped_rows:
            from .grouped_moe_runtime import validate_rows as validate_grouped
            validate_grouped(policy.grouped_rows)
            for rows in policy.grouped_rows:
                self.plan_for(rows)

    def supports_grouped(self, rows):
        return (type(rows) is int and bool(self.grouped_rows) and
                (rows in self.grouped_rows or 505 <= rows <= max(self.grouped_rows)))

    def plan_for(self, rows):
        if type(rows) is not int or not self.supports_grouped(rows):
            raise ConfigError('Unsupported grouped MoE static shapes')
        plan = self.plans.get(rows)
        if plan is None:
            from .grouped_moe_runtime import plan_for as build_plan
            plan = build_plan(rows, min_mme_rows=self.policy.grouped_min_mme_rows,
                              total_slot_budget=self.policy.grouped_slot_budget,
                              total_padded_row_budget=self.policy.grouped_padded_row_budget)
            if not plan.costs(rows, self.top_k, self.num_experts)['fits_budget']:
                raise ConfigError('Grouped MoE workspace exceeds installed budget')
            self.plans[rows] = plan
        return plan

    def select(self, mode, rows):
        if type(rows) is not int or rows < 0:
            raise ConfigError('Static token rows must be a nonnegative integer')
        if mode not in ('auto', 'compact', 'broadcast', 'sorted'):
            raise ConfigError('Unknown MoE mode')
        if mode == 'auto':
            if self.supports_grouped(rows):
                self.plan_for(rows)
                mode = 'grouped'
            elif rows in self.policy.compact_rows or rows in _DIAGNOSTIC_ROWS.get():
                mode = 'compact'
            else:
                mode = 'broadcast'
        counts = self.counts.setdefault(str(rows), {})
        counts[mode] = counts.get(mode, 0) + 1
        return mode

    def snapshot(self):
        return {'compact_rows': list(self.policy.compact_rows),
                'grouped_rows': list(self.policy.grouped_rows),
                'grouped_min_mme_rows': self.policy.grouped_min_mme_rows,
                'grouped_slot_budget': self.policy.grouped_slot_budget,
                'grouped_padded_row_budget': self.policy.grouped_padded_row_budget,
                'mode': 'auto', 'python_capture_calls_by_rows': {k: dict(v) for k, v in self.counts.items()},
                'scope': 'model-owned static metadata at capture; not device invocation counts'}


def dispatch_consumers(model):
    """Preflight the model before draining or retiring any existing captures."""
    setters = []
    for module in model.modules():
        setter = getattr(module, 'set_moe_dispatch_state', None)
        if callable(setter):
            setters.append(setter)
    if not setters:
        raise ConfigError('No model-owned MoE dispatch consumers')
    return tuple(setters)


def install_state(model, policy, *, prepared=None):
    """Caller drains work and invalidates captures before changing ownership."""
    state = DispatchState(policy) if prepared is None else prepared
    if type(state) is not DispatchState or state.policy != policy:
        raise ConfigError('Prepared dispatch differs from installed policy')
    for setter in dispatch_consumers(model):
        setter(state)
    return state


# Explicit compatibility for standalone historical component probes. Serving
# always supplies the state owned by its model; this is not a production selector.
_LEGACY_STATE = DispatchState(MoEDispatch(compact_rows=(1,)))


def set_rows(rows, grouped_rows=()):
    global _LEGACY_STATE
    state = DispatchState(MoEDispatch(compact_rows=rows, grouped_rows=grouped_rows))
    if grouped_rows:
        from .grouped_moe_runtime import prepare
        prepare()
    _LEGACY_STATE = state
    return state.snapshot()


def select_mode(mode, rows):
    return _LEGACY_STATE.select(mode, rows)


def snapshot(state=None):
    return (_LEGACY_STATE if state is None else state).snapshot()


def forward(x, ids, routing, gp, gs, down, ds, table, directions, *, state=None):
    """One production entry; only static metadata selects a graph family."""
    from .batch_ops import moe
    state = _LEGACY_STATE if state is None else state
    rows = int(x.shape[0])
    mode = state.select('auto', rows)
    args = (x, ids, routing, gp, gs, down, ds, table, directions)
    if mode == 'grouped':
        from .grouped_moe_runtime import forward as grouped
        from gaudi_kernels.serving.diagnostics import boundary_hashes as boundary
        return grouped(*args, plan=state.plan_for(rows), use_graph=not boundary.active())
    return moe(*args, mode=mode)

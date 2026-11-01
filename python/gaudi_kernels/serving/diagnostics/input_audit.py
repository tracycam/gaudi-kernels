"""Explicit, opt-in CPU input snapshots for numerical diagnostics."""


class InputAudit:
    def __init__(self):
        self.audit_inputs = None

    def capture(self, role, source):
        if self.audit_inputs is not None:
            self.audit_inputs[role] = source.clone()

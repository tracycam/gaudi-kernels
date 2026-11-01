"""Compatibility entry for the framework-owned reversible CPU diagnostic."""
from gaudi_kernels.serving.diagnostics.host_policy import SchedAttr, configure, submit_threads

__all__ = ['SchedAttr', 'configure', 'submit_threads']

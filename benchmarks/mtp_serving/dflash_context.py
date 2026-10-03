# SPDX-License-Identifier: Apache-2.0
"""Benchmark compatibility import; the implementation is repository-owned serving infrastructure."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.serving.draft.context import CommittedContext  # noqa: F401

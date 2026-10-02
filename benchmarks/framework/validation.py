"""Strict evidence gates shared by framework probes and CPU audit tests."""
import math


def timings_agree(events,walls):
    return bool(events) and len(events)==len(walls) and all(
        math.isfinite(e) and math.isfinite(w) and e>0 and w>0
        and e<=w*1.1+.25 and w<=max(e*1.35,e+2)
        for e,w in zip(events,walls))

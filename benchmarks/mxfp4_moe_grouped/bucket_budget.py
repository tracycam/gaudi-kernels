"""Offline capacity proof for a proposed static Me-bucket graph, not an operator.

Every active expert belongs to exactly one interval [lo, hi]. Since there are at
most T*R routes, at most floor(T*R/lo) experts occupy that interval. Reserving
min(E, floor(T*R/lo)) slots, each hi rows, therefore cannot overflow. The bound
holds jointly without a host read of counts, though the individually maximal
bucket capacities cannot all be occupied at once and waste compute/memory.
"""
import argparse
import json
from pathlib import Path


def buckets(tokens, experts, routes):
    result, lo, hi = [], 1, 16
    while lo <= tokens:
        upper = min(hi, tokens)
        capacity = min(experts, tokens * routes // lo)
        result.append({"min_M_e": lo, "max_M_e": upper,
                       "expert_slots": capacity, "row_slots": capacity * upper})
        lo, hi = hi + 1, hi * 2
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for tokens in [1, 2, 8, 16, 32, 64, 128, 256, 512, 513, 1024]:
        experts, routes = 384, 8
        plan = buckets(tokens, experts, routes)
        slots = sum(b["row_slots"] for b in plan)
        for kind in ["uniform", "hot_expert", "hot_set", "masked"]:
            ids = []
            for token in range(tokens):
                row = [(token * routes + r) % experts for r in range(routes)]
                if kind == "hot_set":
                    row = list(range(routes))
                if kind == "hot_expert":
                    row = [0] + [1 + (token * (routes - 1) + r) % (experts - 1)
                                 for r in range(routes - 1)]
                if kind == "masked":
                    row = [e if (token + r) % 3 else -1 for r, e in enumerate(row)]
                assert len(set(e for e in row if e >= 0)) == sum(e >= 0 for e in row)
                ids.extend(row)
            groups = [[i for i, expert in enumerate(ids) if expert == e]
                      for e in range(experts)]
            occupied, seen, assigned = [0] * len(plan), [], []
            for expert, group in enumerate(groups):
                if not group:
                    continue
                hit = [j for j, b in enumerate(plan)
                       if b["min_M_e"] <= len(group) <= b["max_M_e"]]
                assert len(hit) == 1
                bucket = hit[0]
                occupied[bucket] += 1
                assigned.append(expert)
                seen.extend(group)
            assert all(n <= b["expert_slots"] for n, b in zip(occupied, plan))
            valid = [i for i, expert in enumerate(ids) if expert >= 0]
            assert len(seen) == len(set(seen)) and sorted(seen) == valid
            assert len(assigned) == len(set(assigned))
            cases.append({"T": tokens, "E": experts, "R": routes,
                          "distribution": kind, "active_routes": len(valid),
                          "active_experts": len(assigned), "max_M_e": max(map(len, groups)),
                          "buckets": plan, "occupied_expert_slots": occupied,
                          "static_row_slots": slots, "cap_T_row_slots": experts * tokens,
                          "slot_expansion": slots / len(valid),
                          "active_weight_scale_bytes": len(assigned) * 512 * 6144 * 17 // 32,
                          "all_weight_scale_bytes": experts * 512 * 6144 * 17 // 32,
                          "grouped_A_gp_bytes": slots * 6144 * 2,
                          "grouped_Y_gp_bytes": slots * 512 * 4,
                          "grouped_A_down_bytes": slots * 256 * 2,
                          "grouped_Y_down_bytes": slots * 6144 * 4,
                          "routes_and_experts_assigned_once": True})
    result = {"qualification": "CPU capacity/mapping proof only; bucket device routing and graph not implemented",
              "all_checks_pass": True, "device_validated": False,
              "weight_read_statement": "A proposed counts-gated decoder could read each assigned expert once; no executed bucket kernel or physical HBM measurement",
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "all_checks_pass": True,
                      "T512_row_slots": next(c["static_row_slots"] for c in cases if c["T"] == 512),
                      "device_graph_implemented": False}))


if __name__ == "__main__":
    main()

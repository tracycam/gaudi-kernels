"""Static full-MoE capacities; dynamic route values stay on the device.

This module does not select a production kernel or claim compiler lifetimes.
All byte/FLOP bounds are explicit, including the functional concat-copy bound.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    lower: int
    capacity: int
    experts: int
    slot_base: int
    row_base: int


@dataclass(frozen=True)
class MoePlan:
    mode: str = 'capacity_t'
    gp_expert_batch: int = 2
    down_expert_batch: int = 16
    n_tile: int = 512
    # Conservative safety default for initial public-graph gates. Raising it is
    # explicit; it does not qualify placement or actual peak memory on HPU.
    activation_budget: int = 64 * 1024**2
    caller_certifies_fast_arithmetic: bool = False
    allow_unqualified_current_graph: bool = False

    def buckets(self, tokens, routes, experts):
        if not (2 <= tokens <= 65536 and 1 <= routes <= experts):
            raise ValueError('M1 retains the historical path; require T>=2 and 1<=R<=E')
        if self.mode not in ('capacity_t', 'bucket'):
            raise ValueError('explicit capacity_t or bucket plan required')
        if self.mode == 'capacity_t':
            return [Bucket(1, tokens, experts, 0, 0)]
        result, lo, limit, slot, row = [], 1, 16, 0, 0
        while lo <= tokens:
            cap, count = min(limit, tokens), min(experts, tokens * routes // lo)
            result.append(Bucket(lo, cap, count, slot, row))
            slot, row = slot + count, row + cap * count
            lo, limit = limit + 1, limit * 2
        return result

    def costs(self, tokens, routes, experts, active_experts=None):
        if (self.gp_expert_batch < 1 or self.down_expert_batch < 1
                or self.down_expert_batch % self.gp_expert_batch
                or self.n_tile < 512 or self.n_tile % 512 or 6144 % self.n_tile):
            raise ValueError('aligned tiles and down batch multiple of GP batch required')
        buckets = self.buckets(tokens, routes, experts)
        rows = sum(b.capacity * b.experts for b in buckets)
        slots = sum(b.experts for b in buckets)
        ceil = lambda x, y: (x + y - 1) // y
        gp_fragments = sum(ceil(b.experts, self.gp_expert_batch) for b in buckets)
        down_fragments = sum(ceil(b.experts, self.down_expert_batch) for b in buckets)
        n_tiles = 6144 // self.n_tile
        max_gp_rows = max(b.capacity * min(b.experts, self.gp_expert_batch) for b in buckets)
        max_down_rows = max(b.capacity * min(b.experts, self.down_expert_batch) for b in buckets)
        gp_decoded = min(self.gp_expert_batch, max(b.experts for b in buckets)) * 512 * 6144 * 2
        down_decoded = min(self.down_expert_batch, max(b.experts for b in buckets)) * self.n_tile * 256 * 2
        if max(gp_decoded, down_decoded) > 16 * 1024**2:
            raise ValueError('individual decoded weight fragment exceeds 16 MiB')
        gate = rows * 256 * 2
        tile = rows * self.n_tile * 4
        # Expected serial-fragment bound plus both concat sources/destination.
        # Actual graph scheduling/liveness must still be validated independently.
        activation_bound = (2 * gate + 2 * tile + max_gp_rows * (6144 * 2 + 512 * 4)
                            + max_down_rows * 256 * 2 + 2 * tokens * 6144 * 4)
        if active_experts is not None and not 0 <= active_experts <= min(experts, tokens * routes):
            raise ValueError('active expert count out of range')
        return dict(
            tokens=tokens, routes=routes, experts=experts, mode=self.mode,
            row_slots=rows, expert_slots=slots, buckets=[b.__dict__ for b in buckets],
            gp_mme_nodes=gp_fragments, down_mme_nodes=down_fragments * n_tiles,
            mme_nodes=gp_fragments + down_fragments * n_tiles,
            count_nodes=1, map_nodes=1, local_gp_gather_nodes=gp_fragments,
            tpc_nodes=2 + 3*gp_fragments + down_fragments*n_tiles + n_tiles,
            compute_nodes=2 + 4*gp_fragments + 2*down_fragments*n_tiles + n_tiles,
            gp_gate_nodes=gp_fragments, down_n_tiles=n_tiles,
            requested_MME_FLOPs=2 * rows * (512 * 6144 + 6144 * 256),
            useful_FLOPs_if_all_routes_valid=2 * tokens * routes * (512 * 6144 + 6144 * 256),
            count_ID_scalar_reads=experts * tokens * routes,
            map_count_scalar_reads_upper=slots * experts if self.mode == 'bucket' else experts,
            all_original_weight_scale_bytes=experts * (512 * 6144 + 6144 * 256) * 17 // 32,
            active_original_weight_scale_bytes=(None if active_experts is None else
                active_experts * (512 * 6144 + 6144 * 256) * 17 // 32),
            gp_decoded_fragment_bytes=gp_decoded, down_decoded_fragment_bytes=down_decoded,
            all_decoded_SRAM_write_bytes=slots * (512 * 6144 + 6144 * 256) * 2,
            retained_padded_gate_bytes=gate, padded_down_tile_bytes=tile,
            avoided_full_padded_down_bytes=rows * 6144 * 4,
            desired_activation_upper_bytes=activation_bound,
            configured_activation_budget=self.activation_budget,
            fits_activation_budget=activation_bound <= self.activation_budget,
            compiler_lifetime_and_SRAM_placement_verified=False,
        )

    def require_execution_contract(self, tokens, routes, experts):
        cost = self.costs(tokens, routes, experts)
        if not self.allow_unqualified_current_graph or not self.caller_certifies_fast_arithmetic:
            raise ValueError('experimental current-graph opt-in and fast arithmetic certificate required')
        if not cost['fits_activation_budget']:
            raise ValueError('functional MoE activation bound exceeds configured budget')
        return cost

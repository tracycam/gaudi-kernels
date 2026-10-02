"""Frozen70-g label aliases, used only by validation tools."""
import json
from gaudi_kernels.engine.config import ConfigError, EngineConfig
from gaudi_kernels.engine.selection import PolicySelection as TypedSelection

class PolicySelection(TypedSelection):
    @property
    def features(self):
        d = self.engine.decode
        result = []
        if d.moe.gate_up.impl != 'legacy':
            result.append('gp_fold')
        if d.moe.gate_up.impl == 'tpc_folded_scale_tail':
            result.append('gp_scale_tail')
        if d.moe.down.impl == 'tpc_vector':
            result.append('down_vec')
        if d.attention.swa != 'vendor':
            result.append('swa_' + d.attention.swa)
        if d.qkv.reduce == 'neumaier_isa':
            result.append('qkv_neumaier_isa')
        if d.qkv.post != 'vendor':
            result.append('qkv_post')
        if d.tp_reduce.collective == 'all_gather_fp32_local_sum':
            result.append('f32_ag')
        if d.moe.router.post == 'vector_top8':
            result.append('router_post_vector')
        if d.norm.impl != 'vendor':
            result.append('norm_fp32')
        compact = d.moe.dispatch.compact_rows
        if compact == tuple(range(1, 9)):
            result.append('moe_compact8')
        elif compact == (1, 8):
            raise ConfigError('The pinned70-g installer predates compact-only1/8; migrate and qualify that installer first')
        elif compact != (1,):
            # The old baseline always used compact M=1. Empty selectors need
            # an explicit installer, not a mislabeled legacy baseline.
            raise ConfigError('Legacy installer cannot express compact_rows=[]')
        if d.moe.combine.sum_dtype == 'bf16':
            result.append('moe_sum_bf16')
        if d.norm.impl == 'fp32_grid24_quant':
            result.append('norm_qkv_grid24')
        if d.qkv.post == 'fused_all':
            result.append('qkv_post_full')
        return tuple(result)

    @property
    def label(self):
        return '+'.join((self.block_policy, *self.features))


def qualified_selections():
    control = EngineConfig()
    full_doc = control.to_dict()
    full_doc['decode']['qkv']['post'] = 'fused_all'
    full = EngineConfig.from_dict(full_doc)
    broadcast_doc = full.to_dict()
    broadcast_doc['decode']['moe']['dispatch']['compact_rows'] = [1]
    broadcast = EngineConfig.from_dict(broadcast_doc)
    reference = {
        'decode': {'qkv': {'impl': 'block_fp8_a16', 'a8_max_rows': 0,
                           'reduce': 'sequential', 'post': 'vendor'},
                   'norm': {'impl': 'fp32'}, 'attention': {'swa': 'vendor'},
                   'moe': {'router': {'post': 'vendor'}, 'gate_up': {'impl': 'legacy'},
                           'down': {'impl': 'legacy'}, 'combine': {'sum_dtype': 'fp32'},
                           'dispatch': {'compact_rows': [1]}},
                   'tp_reduce': {'collective': 'baseline'}}}
    selections = [PolicySelection(c) for c in (control, full, broadcast, EngineConfig.from_dict(reference))]
    # CPU references used by the original 70-g arithmetic/quality gates.
    for swa, collective in [('vendor', 'baseline'),
                            ('fp32_fast', 'all_gather_fp32_local_sum')]:
        document = json.loads(json.dumps(reference))
        document['decode']['attention']['swa'] = swa
        document['decode']['tp_reduce']['collective'] = collective
        for impl in ('cpu_ocp_a8_bf16_fp32', 'cpu_native_a8_bf16_fp32',
                     'cpu_ocp_a8_fp32_v1', 'cpu_native_a8_fp32_v1'):
            selections.append(PolicySelection(EngineConfig.from_dict(document), impl))
    table = {selection.label: selection.to_dict() for selection in selections}
    # Historical diagnostics spell these three independent options in a
    # different order. Accept only these exact saved labels at the benchmark
    # boundary; the worker still receives the same structured choices.
    for selection in selections:
        if selection.reference != 'device' and selection.engine.decode.attention.swa == 'fp32_fast':
            table[selection.reference + '+norm_fp32+swa_fp32_fast+f32_ag'] = selection.to_dict()
    return table



def typed_policy(label):
    table=qualified_selections()
    if label not in table:raise ValueError("Unqualified frozen70-g label")
    return table[label]

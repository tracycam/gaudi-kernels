from gaudi_kernels.engine.context import context as execution_context
"""Explicit prior-qualification identities, not an env-controlled approval list.

These are the exact public block-FP8 and reducer binaries sealed in 70-c and
the referenced standalone numerical/ISA gates. A rebuild is unknown until a
separate qualification updates this versioned source. No library is loaded here.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

_CORE = {
    ('block_fp8', 'tpc'): '65fe77190fd6e62699800736b45c3f6190ce233f2aa933b5fdf09f1f6941a977',
    ('block_fp8', 'torch'): '3c9a092b4461c0990200cb88fb04aab62afe031b4e3b0ce488d0177aac6c6e0a',
}
# Separate opt-in identity, qualified by complete operator old/new bit gates,
# actual deployed ELF simulation and unchanged executable/data sections of
# quant/decode_fast/reduce/finish. Compiler hashedISA/path metadata differs;
# do not claim the full embedded ELFs are identical. Model acceptance remains
# a later gate. Arbitrary rebuilt libraries are still rejected.
_NATIVE_LANES_TPC = '121173a4dca0c293db042f904cdb16d2d74ab58022a9cb2f76d9a4785b776740'
_NATIVE_LANES_REDUCE_ELF = '61180a5dc37334385bbecf817b7b0f36aecbf06e03e4bb14db6a747077762776'
_REDUCERS = {
    'sequential': dict(operator='gaudi_block_fp8::reduce', libraries={},
        elf_sha256='71c96408749d1bf0c04067034a1fcbeee3a679bb11f461148427e33464a38fc7',
        text_sha256='45bb211d6e5475e66716e3771d18f84e7d226800d60d9945307cd92604253416',
        qualification='docs/BLOCK-FP8-PRODUCTION-HOOK.md'),
    'neumaier_fp32': dict(operator='gk_reduce_experiment::neumaier', libraries={
        ('reduce', 'tpc'): '3c6c11151bc5e5e589daa1e006493e85168f4ce040a5b1803121498f244fec5e',
        ('reduce', 'torch'): '4ea7ce7925c66a484167b0a3679e9475f7abf95d52ecbbd6ce7e762a4718b5e4'},
        elf_sha256='a48ef0e603a43b177fcaeb823970ab2a629b963d9e7879ddee303f84f9740bbb',
        text_sha256='86228b24fa6c4a8d3a363a7f10b2c8d70d57d022d7a0fcfcce869eb32b8801d9',
        qualification='docs/BLOCK-FP8-REDUCE-CANDIDATES-20260928.md'),
    'neumaier_isa_fp32': dict(operator='gk_reduce_isa::handschedule', libraries={
        ('reduce_isa', 'tpc'): 'b60f5fbc597b7314653802d3ecf618d9306349f04b62054242ec8232bee5305a',
        ('reduce_isa', 'torch'): '62e4f3eaae0ad25c5d370bf007ffe9fe7dea7f69bba5b58a18b0c64735d6ecaa'},
        elf_sha256='c1d5c7b06ed7e3c769be43e453e28d7a9e5bfbb322695dda4fe52fee0e997002',
        text_sha256='b951d287c14e41c6036002565710ce0903734a8fd6d007c09e2dc7921724e06b',
        qualification='docs/NEUMAIER-ISA-20260928.md'),
}


@dataclass(frozen=True)
class _VerifiedImplementation:
    reduction: str
    runtime_loaded: bool
    record: dict


def verify_artifacts(reduction, *, require_loaded=True):
    import torch
    if reduction not in _REDUCERS:
        raise ValueError('unqualified FP32 reduction algorithm')
    entry = dict(_REDUCERS[reduction])
    variant = execution_context().selection.engine.decode.qkv.core
    core = dict(_CORE)
    if variant == 'native_lanes_v1':
        core[('block_fp8', 'tpc')] = _NATIVE_LANES_TPC
        if reduction == 'sequential':
            entry['elf_sha256'] = _NATIVE_LANES_REDUCE_ELF
    elif variant != 'linear_scale_v1':
        raise ValueError('unqualified block-FP8 core variant')
    records = {}
    loaded = {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    for key, expected in {**core, **entry['libraries']}.items():
        path = Path(execution_context().path(*key)).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('unknown FP32 implementation bytes: ' + str(key))
        if key[1] == 'torch' and require_loaded and str(path) not in loaded:
            raise ValueError('qualified Torch implementation has not been loaded: ' + str(key))
        if key[1] == 'tpc' and require_loaded:
            database = {str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p}
            if str(path) not in database:
                raise ValueError('qualified TPC database is absent from GC_KERNEL_PATH: ' + str(key))
        records['.'.join(key)] = dict(path=str(path), sha256=actual)
    return _VerifiedImplementation(reduction, require_loaded,
        dict(contract='fp32_implementation_binding_v1', operator=entry['operator'],
             core_variant=variant,
             reducer_elf_sha256=entry['elf_sha256'], reducer_text_sha256=entry['text_sha256'],
             qualification=entry['qualification'], libraries=records,
             scope='exact prior-qualified implementation and loaded public operator; not full-model acceptance'))

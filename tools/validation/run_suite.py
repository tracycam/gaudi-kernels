#!/usr/bin/env python3
"""Run canonical repository source through the bounded TP8 validation suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.validation.prepare_canonical import prepare
from tools.validation.legacy_selection import PolicySelection
from gaudi_kernels.engine import EngineConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--replay-library',type=Path,required=True)
    p.add_argument('--plugin-source',type=Path,help='Repository-owned plugin checkout with explicit adapter boundary')
    p.add_argument('--preflight-post',choices=('fused_swa','fused_all'),default='fused_all',
                   help='Reproduce the original SWA-only failure without broadening attention scope')
    p.add_argument('--preflight',action='store_true');p.add_argument('--setup-reference',action='store_true');p.add_argument('--omit-boundary-check',action='store_true');p.add_argument('--python',default=sys.executable)
    p.add_argument('--numa-binding',choices=('none','local'),default='none')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if a.preflight_post=='fused_swa' and (not a.preflight or not a.omit_boundary_check):
        p.error('SWA-only isolation requires --preflight --omit-boundary-check; all-layer hash gates require fused_all')
    a.output_root.mkdir(parents=True,exist_ok=True)
    manifest=a.output_root/(a.case+'.manifest.json')
    prepare(a.control,manifest,a.output_root/a.case,a.replay_library,2 if a.preflight else 70)
    document=json.loads(manifest.read_text())
    document['selection']['engine']['runtime']['host']={'numa_binding':a.numa_binding,'visible_modules':list(range(8))}
    manifest.write_text(json.dumps(document,indent=2)+'\n')
    baseline=json.loads((a.control/'baseline.json').read_text())
    if a.preflight:
        candidate=PolicySelection(EngineConfig.from_dict({'decode':{'qkv':{'post':a.preflight_post}}})).label
        reference='bf16_fp32+norm_fp32'
        arguments=['--timeout-seconds','1800','--layers','2','--tokens','8','--context','512',
                   '--integration-preflight','--single-rpc','--reuse-pages','--cpu-oracle',
                   '--production-policies',reference+','+candidate,'--quality-contract','fp32_arithmetic_v1',
                   '--quality-reference-policy',reference,'--quality-candidate-policies',candidate]
    else:
        arguments=baseline['historical_command'][4:]
        arguments[arguments.index('--batch-sizes')+1]='1,2,3,8'
    if not a.omit_boundary_check:arguments.append('--boundary-hashes')
    if a.setup_reference:arguments.append('--setup-reference')
    command=[a.python,'-m','tools.validation.executor.run_model',a.case,'native',
             '--output-root',str(a.output_root),'--manifest',str(manifest),
             '--plugin-source',str(a.plugin_source or a.control/'runtime/plugin'),'--vllm-source',str(a.control/'runtime/vllm'),*arguments]
    (a.output_root/(a.case+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
    raise SystemExit(subprocess.call(command,cwd=root))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Full QKV CPU cost and both trees; synthetic evidence is not device proof."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from gaudi_kernels.block_fp8_fp32_contract import build_reference, audit, epilogue_certificate, require_fast_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    inputs=torch.load(args.inputs,weights_only=False,map_location='cpu')
    backend=require_fast_backend()
    start=time.monotonic();ref=build_reference(**inputs);reference_s=time.monotonic()-start
    actual={k:ref[k].clone() for k in ('q_native','activation_scales','prepared_weight','prepared_scales','bias')}
    actual['partial']=ref['partial_balanced'].clone()
    start=time.monotonic()
    actual['output']=epilogue_certificate(actual['partial'].numpy()[:,0,:],ref['factors'],ref['bias'].numpy(),'neumaier_fp32')
    epilogue_s=time.monotonic()-start
    start=time.monotonic();report=audit(ref,actual,reduction='neumaier_fp32');audit_s=time.monotonic()-start
    assert report['numerical_passed'] and not report['passed']
    torch.save(actual,args.out/'synthetic_staged_evidence.pt')
    result=dict(status='PASS_CPU_REFERENCE_ONLY',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        shape=ref['shape'],cpu_threads=1,reference_s=reference_s,epilogue_s=epilogue_s,audit_s=audit_s,
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),cpu_backend=backend,report=report,
        independent_tree_bf16_differences=int((ref['outputs']['balanced']!=ref['outputs']['sequential']).sum()),
        scope='synthetic CPU arithmetic and full-shape wall cost; no device/model acceptance; no performance claim')
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('report','cpu_backend')},indent=2))

if __name__=='__main__':main()

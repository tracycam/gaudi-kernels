"""Bind the owned grouped MoE provider and typed prefill size to a manifest.

Only deployment bindings change. No source files are rewritten, no device is
acquired, and main/installed vLLM files are never modified.
"""
import argparse
import json
from pathlib import Path
from gaudi_kernels.engine.artifact_pins import PINS
from gaudi_kernels.engine.context import initialize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--layers',type=int,choices=(2,70),default=70)
    p.add_argument('--chunk',type=int,choices=(512,2048,4096),required=True)
    p.add_argument('--rows',type=int,nargs='+',required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    document=json.loads(a.base.read_text())
    document['startup'].update(run_dir=str(a.run_dir.resolve()),layers=a.layers)
    document['artifacts'].pop('batch.tpc',None)
    for key,name in (('moe_bundle.tpc','libgaudi_moe_serving_tpc.so'),
                     ('expert.torch','gaudi_expert_partition.so'),
                     ('moe_batch_provider.host','provider_batch.so'),
                     ('moe_expert_provider.host','provider_expert.so')):
        document['artifacts'][key]={'path':str((a.bundle/name).resolve()),'sha256':PINS[key]}
    cfg=document['selection']['engine']
    cfg['decode']['moe']['dispatch']['grouped_rows']=a.rows
    cfg['runtime']['prefill_chunk_tokens']=a.chunk
    initialize(document,directory=a.base.resolve().parent)
    a.output.write_text(json.dumps(document,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()

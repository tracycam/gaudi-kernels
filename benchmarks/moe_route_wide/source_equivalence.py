"""Prove cloned guards differ only by declared namespace/domain substitutions."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2]
changes={
'csrc/host/moe_route_metadata_v3_glue.cpp':('csrc/host/route_wide_metadata_glue.cpp',[('gk_route_','gk_route_wide_'),('c<=32','c<=128'),('513*8*32','513*8*128')]),
'csrc/host/moe_route_tiles_glue.cpp':('csrc/host/route_wide_tiles_glue.cpp',[('gk_route_tile_','gk_route_wide_tile_'),('p[1]<=32','p[1]<=128'),('c<=32','c<=128'),('513*8*32','513*8*128')]),
'csrc/torch/moe_route_metadata_v3.cpp':('csrc/torch/route_wide_metadata.cpp',[('gaudi_route_metadata_v3','gaudi_route_metadata_wide'),('gk_route_','gk_route_wide_'),('c<=32','c<=128')]),
'csrc/torch/moe_route_tiles.cpp':('csrc/torch/route_wide_tiles.cpp',[('gaudi_route_tiles','gaudi_route_wide_tiles'),('gk_route_tile_','gk_route_wide_tile_'),('c<=32','c<=128'),('p.size(1)<=32','p.size(1)<=128'),('513*8*32','513*8*128')])}
records=[]
for src,(dst,substitutions)in changes.items():
 old=subprocess.check_output(['git','show','822e049:'+src],cwd=root);assert(root/src).read_bytes()==old,'default source changed'
 expected=old.decode()
 for x,y in substitutions:assert x in expected;expected=expected.replace(x,y)
 prefix='// PRIVATE wide-row experiment: copied from '+src+' at 822e049.\n// Only namespace/GUID and explicit row limit change; original interface is untouched.\n'
 assert(root/dst).read_text()==prefix+expected,dst
 records.append(dict(original=src,new=dst,original_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256((root/dst).read_bytes()).hexdigest(),substitutions=substitutions))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='PASS_EXACT_DECLARED_INTERFACE_DIFF',base='822e049',records=records),indent=2)+'\n')

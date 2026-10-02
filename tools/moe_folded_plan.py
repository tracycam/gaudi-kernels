"""Write/read a closed library inventory for the four-way folded ablation."""
import argparse
import json
from pathlib import Path
from moe_activation_fold_core import embedded_elf,sha,verify_deployed

BRIDGES={'gp':('gaudi_gp_diagnostic.so','fc403180e48a6a750f963743f878b11384a0313ac7919ce32bc8e4ac874b7895'),
         'down':('gaudi_down_activation.so','8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d')}
ELFS={'gp':'f4cacc2a759bdae8936e98971af8e5035c6f0653261ceb56b5a6fe554c831d17',
      'down':'13a21c88858a565bfe62889ae556a4f53aa866145a7ce11561d25ffcb71b765b'}

def validate(path):
    plan=json.loads(path.read_text())
    assert plan['format']=='gk-folded-plan-v1'
    for record in plan['files'].values():
        target=Path(record['path']).resolve(strict=True)
        assert sha(target.read_bytes())==record['sha256'],target
    for kind in ('gp','down'):
        verify_deployed(Path(plan['files'][kind+'_torch']['path']).parent,kind)
        assert plan['files'][kind+'_torch']['sha256']==BRIDGES[kind][1]
    data=Path(plan['files']['folded_tpc']['path']).read_bytes()
    for kind,digest in ELFS.items():assert sha(embedded_elf(data,'folded_'+kind))==digest
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gp-dir',type=Path,required=True);p.add_argument('--down-dir',type=Path,required=True)
    p.add_argument('--folded-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan={'format':'gk-folded-plan-v1','files':{},'candidate_embedded_elf_sha256':ELFS,'default_changed':False}
    def add(name,path):plan['files'][name]={'path':str(path.resolve(strict=True)),'sha256':sha(path.read_bytes())}
    for kind,directory in [('gp',a.gp_dir),('down',a.down_dir)]:
        data,proof=verify_deployed(directory,kind);add(kind+'_torch',directory/BRIDGES[kind][0]);plan[kind+'_deployed']=proof
    build=json.loads((a.folded_dir/'build.json').read_text())
    assert build['committed_sources_verified'] and build['output_embedded_elfs_verified']
    plan['build_source_commit']=build['source_commit']
    for kind,name in [('folded_torch','gaudi_moe_activation_folded.so'),('folded_tpc','libgaudi_moe_activation_folded_tpc.so')]:
        add(kind,a.folded_dir/name);assert plan['files'][kind]['sha256']==build['files_sha256'][name]
    a.output.write_text(json.dumps(plan,indent=2)+'\n');validate(a.output)
    print(json.dumps({'status':'PINNED_PLAN_VERIFIED','path':str(a.output)}))

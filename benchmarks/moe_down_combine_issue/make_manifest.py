"""CPU-only actual-byte manifest; candidate precision/performance still unqualified."""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from moe_activation_fold_core import embedded_elf
paths={'gp_torch':'runtime/gaudi_gp_scale_tail.so','gp_tpc':'runtime/libgaudi_gp_scale_tail_tpc.so','down_torch':'runtime/gaudi_down_activation.so','down_tpc':'runtime/libgaudi_down_activation_tpc.so','batch_tpc':'fixtures/executor-a/tpc/libbatch_tpc.so','batch_torch':'fixtures/executor-a/ops-build/unified_batch_ops.so','precision_tpc':'fixtures/executor-a/precision-tpc/libprecision_tpc.so','precision_torch':'fixtures/executor-a/precision-ops-build/precision_ops.so','candidate_tpc':'builds/tpc/libgaudi_down_combine_isa_tpc.so','candidate_torch':'builds/torch/gaudi_down_combine_isa.so'}
pins={'gp_torch':'c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410','gp_tpc':'f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07','down_torch':'8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d','down_tpc':'9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a'}
files={k:dict(path=v,sha256=hashlib.sha256((root/v).read_bytes()).hexdigest()) for k,v in paths.items()}
for key,sha in pins.items():assert files[key]['sha256']==sha,key
build=json.loads((root/'builds/torch/build.json').read_text());assert build['state']=='BUILT_CPU_META_PASS' and not build['offline_Meta_stub'];assert build['library_sha256']==files['candidate_torch']['sha256']
texts={'down_p4':'3427f45bc443045efca30f8594cb5cece8e4b94d9d443e6d6f12dac1ee083029','down_p6':'1df5461a4e933b6d686e339deae62d352aad6480f270f94f1ae942b790ac7160','down_old':'49354a722ed0bcf6fb2a5bcd4e5d74f6c818716fe8ce3f61c97417b5183b88d5'}
elfs={}
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder)
    for name,want in texts.items():
        elf=embedded_elf((root/paths['candidate_tpc']).read_bytes(),name);path=folder/(name+'.o');path.write_bytes(elf);txt=folder/(name+'.text')
        subprocess.run(['objcopy','--dump-section','.text='+str(txt),str(path)],check=True,timeout=20)
        got=hashlib.sha256(txt.read_bytes()).hexdigest();assert got==want,name
        md=folder/(name+'.metadata');subprocess.run(['objcopy','--dump-section','.tpc_metadata='+str(md),str(path)],check=True,timeout=20)
        resource=md.read_bytes();assert int.from_bytes(resource[:4],'little')==20 and resource[4]==1,(name,'lookup resource undeclared')
        elfs[name]=dict(specialFunctionUsed=resource[4],elf_sha256=hashlib.sha256(elf).hexdigest(),text_sha256=got)
assert elfs['down_old']['elf_sha256']=='b930eb299830f036be389bb4a9d7f0597fa9ee24e51c2880e4e308689c6d3b75'
identity=json.loads((root/'source-identity.json').read_text())
for name,sha in identity['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
result=dict(source_commit=identity['git_commit'],files=files,elfs=elfs,offline_Meta_stub=False,executor_fixture='fixtures/executor-a',kernel_library_keys=['batch_tpc','precision_tpc','gp_tpc','down_tpc','candidate_tpc'],device_qualified=False,model_quality_qualified=False,scope='Target CPU link/Meta and actual library/text identity; device qualification pending')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

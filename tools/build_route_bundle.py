"""Combine pinned qualified ELF bytes and unchanged glue; never compile TPC math."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
for role in ('core','metadata','tiles'):p.add_argument('--'+role+'-library',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--abi-include',default='/usr/lib/habanatools/include')
p.add_argument('--pins',type=Path,default=root/'benchmarks/route_bundle/qualified_inputs.json')
p.add_argument('--row-first',action='store_true',help='requires separately qualified row-first pins and matching glue macro')
p.add_argument('--duplicate-test',action='store_true',help='CPU rejection test only; never load on a device')
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((root/'source-identity.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for name,want in identity['files_sha256'].items():
    if sha(root/name)!=want:raise RuntimeError('frozen source changed: '+name)
pins=json.loads(a.pins.read_text());(out/'source-identity.json').write_bytes((root/'source-identity.json').read_bytes())
required_tile_defines=['GK_ROUTE_GATHER_ROW_FIRST=1']if a.row_first else []
if pins['tiles'].get('glue_defines',[])!=required_tile_defines:raise RuntimeError('pins and --row-first disagree')
meta=dict(source_commit=identity['git_commit'],state='building',TPC_compiler_invoked=False,
          duplicate_test=a.duplicate_test,row_first=a.row_first,pins=pins,commands=[],elfs={},libraries={},
          sdk_header_sha256=sha(Path(a.abi_include)/'tpc_kernel_lib_interface.h'))
def save():(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
def run(command,log):
    meta['commands'].append(command);save();log.write(json.dumps(command)+'\n');log.flush()
    subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
sources={'core':'mxfp4_moe_graph_glue.cpp','metadata':'moe_route_metadata_v3_glue.cpp','tiles':'moe_route_tiles_glue.cpp'}
apis=('GetKernelGuids','InstantiateTpcKernel','GetShapeInference','GetSupportedDataLayout','GetLibVersion')
try:
    with(out/'build.log').open('w')as log:
        meta['compiler_version']=subprocess.check_output(['g++','--version'],text=True)
        objects=[]
        for role,source in sources.items():
            library=getattr(a,role+'_library').resolve();digest=sha(library)
            if digest!=pins[role]['library_sha256']:raise RuntimeError('unqualified '+role+' library SHA')
            meta['libraries'][role]=dict(source=str(library),sha256=digest)
            handle=ctypes.CDLL(str(library),mode=ctypes.RTLD_LOCAL)
            for name,expected in pins[role]['elfs'].items():
                begin=ctypes.addressof(ctypes.c_ubyte.in_dll(handle,'_binary_'+name+'_o_start'))
                end=ctypes.addressof(ctypes.c_ubyte.in_dll(handle,'_binary_'+name+'_o_end'))
                if not 0<end-begin<4*1024*1024:raise RuntimeError('ELF symbol span')
                data=ctypes.string_at(begin,end-begin)
                if len(data)!=expected['bytes']or hashlib.sha256(data).hexdigest()!=expected['elf_sha256']:
                    raise RuntimeError('qualified embedded ELF mismatch: '+name)
                path=out/(name+'.o');path.write_bytes(data);meta['elfs'][name]=expected
                run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'],log)
                objects.append(name+'_x86.o')
            src=root/'csrc/host'/source;shutil.copy2(src,out/(role+'_original_glue.cpp'))
            command=['g++','-O2','-std=c++17','-fPIC','-fvisibility=hidden','-I'+a.abi_include]
            command+=['-D'+api+'=gk_bundle_'+role+'_'+api for api in apis]
            for define in pins[role].get('glue_defines',[]):
                if define.split('=')[0]not in src.read_text():raise RuntimeError('glue does not implement pinned macro '+define)
                command+=['-D'+define]
            command+=['-c',str(src),'-o',role+'_glue.o'];run(command,log);objects.append(role+'_glue.o')
        if len(meta['elfs'])!=14:raise RuntimeError('exactly fourteen distinct ELF names required')
        export='{global: '+ '; '.join(apis)+'; local: *;};\n';(out/'exports.map').write_text(export)
        wrapper=root/'csrc/host/route_bundle_glue.cpp';shutil.copy2(wrapper,out/'route_bundle_glue.cpp')
        run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
             '-Wl,--version-script=exports.map','-I'+a.abi_include,
             *(['-DGK_ROUTE_BUNDLE_DUPLICATE_TEST=1']if a.duplicate_test else []),
             str(wrapper),*objects,'-o','libgaudi_route_bundle_tpc.so'],log)
        run(['nm','-D','--defined-only','libgaudi_route_bundle_tpc.so'],log)
    meta.update(state='CPU_BUILT_NOT_DEVICE_VALIDATED',artifacts={str(p.relative_to(out)):sha(p)for p in out.iterdir()if p.is_file()and p.name!='build.json'})
except Exception as error:
    meta.update(state='FAILED',error=repr(error));save();raise
save();print(json.dumps(dict(state=meta['state'],sha256=sha(out/'libgaudi_route_bundle_tpc.so'))))

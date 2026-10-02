"""Combine the pinned M1 AV and multirow SWA ELFs without invoking TPC clang."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--av-library', type=Path, required=True)
    p.add_argument('--batch-library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--abi-include', default='/usr/lib/habanatools/include')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    a.out.mkdir(parents=True, exist_ok=False)
    pins = {'av':'24d5a035a7f789bdb30b7b0628d0b735b9044f947566e29ea0d1a439ac45c288',
            'batch':'d0f41724633602ed1b09d346f1cc2f4e711ddfd2c5a40e0a7f719a1e6e96438f'}
    names = {'av':('quad','group','quad_debug','group_debug'), 'batch':('head_batch','head_batch_quad')}
    report = {'status':'BUILDING', 'TPC_compiler_invoked':False, 'inputs':{}, 'elfs':{}, 'commands':[], 'sources':{}}
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        (a.out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(cmd):
        report['commands'].append(cmd);save()
        subprocess.run(cmd,cwd=a.out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    apis = ('GetKernelGuids','InstantiateTpcKernel','GetShapeInference','GetSupportedDataLayout','GetLibVersion')
    try:
        with (a.out/'build.log').open('w') as log:
            objects=[]
            for role in names:
                lib=getattr(a,role+'_library').resolve(strict=True)
                if digest(lib)!=pins[role]:
                    raise ValueError('Unqualified SWA provider: '+role)
                report['inputs'][role]={'path':str(lib),'sha256':pins[role]}
                handle=ctypes.CDLL(str(lib),mode=ctypes.RTLD_LOCAL)
                for name in names[role]:
                    begin=ctypes.addressof(ctypes.c_ubyte.in_dll(handle,'_binary_'+name+'_o_start'))
                    end=ctypes.addressof(ctypes.c_ubyte.in_dll(handle,'_binary_'+name+'_o_end'))
                    if not 0<end-begin<4*1024*1024:
                        raise ValueError('ELF symbol span')
                    elf=ctypes.string_at(begin,end-begin)
                    (a.out/(name+'.o')).write_bytes(elf)
                    report['elfs'][name]={'bytes':len(elf),'sha256':hashlib.sha256(elf).hexdigest()}
                    run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
                    objects.append(name+'_x86.o')
                source=root/'csrc/host'/('swa128_avgroup_glue.cpp' if role=='av' else 'swa128_batch_glue.cpp')
                report['sources'][str(source.relative_to(root))]=digest(source)
                run(['g++','-O2','-std=c++17','-fPIC','-fvisibility=hidden','-I'+a.abi_include,
                     *['-D'+api+'=gk_bundle_'+role+'_'+api for api in apis],
                     '-c',str(source),'-o',role+'_glue.o'])
                objects.append(role+'_glue.o')
            # Reuse the checked provider registry and SDK capacity handling.
            wrapper=(root/'csrc/host/route_bundle_glue.cpp').read_text()
            report['sources']['csrc/host/route_bundle_glue.cpp']=digest(root/'csrc/host/route_bundle_glue.cpp')
            for old,new in [('DECLARE(core) DECLARE(metadata) DECLARE(tiles)','DECLARE(av) DECLARE(batch)'),
                ('gk_bundle_core_','gk_bundle_av_'),
                ('PROVIDER(core,7),PROVIDER(metadata,4),PROVIDER(tiles,3)','PROVIDER(av,4),PROVIDER(batch,2)'),
                ('std::array<GuidInfo,14>','std::array<GuidInfo,6>'),
                ('std::array<unsigned,14>','std::array<unsigned,6>'),('owner<3','owner<2')]:
                if old not in wrapper:
                    raise ValueError('Registry template changed: '+old)
                wrapper=wrapper.replace(old,new)
            (a.out/'swa_bundle.cpp').write_text(wrapper)
            (a.out/'exports.map').write_text('{global: '+'; '.join(apis)+'; local: *;};\n')
            run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
                 '-Wl,--version-script=exports.map','-I'+a.abi_include,
                 'swa_bundle.cpp',*objects,'-o','libgaudi_swa_bundle_tpc.so'])
            report.update(status='BUILT_NOT_DEVICE_QUALIFIED',
                library_sha256=digest(a.out/'libgaudi_swa_bundle_tpc.so'))
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        report['files_sha256']={f.name:digest(f) for f in a.out.iterdir() if f.is_file() and f.name!='build.json'}
        save()
    print(json.dumps({'status':report['status'],'sha256':report['library_sha256']}))


if __name__=='__main__':
    main()

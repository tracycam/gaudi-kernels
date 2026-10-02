"""Compile owned native graph and SCAL submission probe; no device acquisition."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];src=Path(__file__).resolve().parent
commands=[['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',kind,str(src/'affine.c'),'-o','affine.'+suffix]for kind,suffix in (('-c','o'),('-S','s'))]
commands.extend([['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','affine.o','affine_x86.o'],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(src/'glue.cpp'),'affine_x86.o','-o','libnative_graph_tpc.so'],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror',str(src/'scal_scope.cpp'),'-ldl','-pthread','-o','libscal_scope.so'],
 ['g++','-O2','-std=c++17','-Wall','-Wextra','-I/usr/include/habanalabs',str(src/'native.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-ldl','-pthread','-o','native']])
commands.extend([
 ['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(root/'csrc/native_graph/graph.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-pthread','-o','libgkg_graph.so'],
 ['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(src/'owned_probe.cpp'),'-L.','-Wl,-rpath,$ORIGIN','-lgkg_graph','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-pthread','-o','owned_probe'],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(root/'csrc/native_graph/capture.cpp'),'-L.','-Wl,-rpath,$ORIGIN','-lgkg_graph','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-ldl','-pthread','-o','libgkg_capture.so'],
 ['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(src/'capture_probe.cpp'),'-L.','-Wl,-rpath,$ORIGIN','-lgkg_capture','-lgkg_graph','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-pthread','-o','capture_probe'],
 ['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(src/'binding_probe.cpp'),'-L.','-Wl,-rpath,$ORIGIN','-lgkg_graph','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-pthread','-o','binding_probe'],
 ['g++','-O2','-ffp-contract=off','-std=c++17','-Wall','-Wextra','-Werror','-I/usr/include/habanalabs',str(src/'comm_probe.cpp'),'-L.','-Wl,-rpath,$ORIGIN','-lgkg_graph','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-lhcl','-pthread','-o','comm_probe']])
meta=dict(commands=commands,device_verified=False,state='BUILDING')
try:
 with(out/'build.log').open('w')as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
  with(out/'affine.dis').open('w')as dis:subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','affine.o'],cwd=out,stdout=dis,stderr=log,check=True)
 meta['state']='COMPILED'
finally:
 meta['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()}
 (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')

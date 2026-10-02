"""Build only; device access is exclusively through the shared probe runner."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from generate import generate, replace_once
from prefetch import transform
from decoder_window import specialize
from decoder_schedule import transform as schedule_decode
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--rows',type=int,choices=[1,2,4],required=True)
p.add_argument('--window',type=int,choices=[0,2,4,8,16,32],default=32)
p.add_argument('--activation',choices=['vector','pair','inc'],default='vector')
p.add_argument('--lookup-x2',action='store_true')
p.add_argument('--n512-window',type=int,choices=[2,4,8,32])
p.add_argument('--literal',choices=['control','prefetch','safe'])
p.add_argument('--decode-canonical-zero',action='store_true',
               help='Private finite-MAC decoder: permit transient -0 to become +0; original storage unchanged')
p.add_argument('--decode-window',type=int,choices=[4,8,16,32],
               help='Grouped-only K32 specialization; no ragged tail, independent decode window')
p.add_argument('--decode-schedule',choices=['standard','increment','pipeline'],default='standard')
p.add_argument('--include-dir',default='/usr/include/habanalabs');a=p.parse_args()
if a.activation in ('pair','inc')and not a.window:p.error('pair prefetch requires explicit K window')
if a.window==2 and a.activation!='vector':p.error('K2 window requires vector activations')
if a.n512_window and (a.rows!=1 or a.lookup_x2 or a.activation!='vector'):p.error('N512 prototype currently supports M1 vector/LUT only')
if a.literal and (a.rows!=1 or a.n512_window or a.lookup_x2 or a.activation!='vector'):p.error('literal GP requires M1 and its fixed layout')
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);generate(root,out,a.activation,a.lookup_x2)
if a.decode_canonical_zero:
 path=out/'decode.c'
 path.write_text(replace_once(path.read_text(),'''    bool128 nonzero=v_bf16_cmp_neq_b(q,0);
    return v_bf16_mul_vb(q,(bfloat128)((exponent&255)<<7),0,q,nonzero);''','''    // Finite-MAC-only optimization: preserve values, not transient zero signs.
    return v_bf16_mul_b(q,(bfloat128)((exponent&255)<<7));'''))
if a.decode_window:
 if a.lookup_x2:p.error('decode-window is measured separately from lookup-x2')
 path=out/'decode.c';path.write_text(specialize(path.read_text(),a.decode_window))
if a.decode_schedule!='standard':
 if a.decode_window!=8:p.error('address/pipeline experiment requires K8 specialization')
 path=out/'decode.c';path.write_text(schedule_decode(path.read_text(),a.decode_schedule))
if a.n512_window:
 source=(root/'benchmarks/mxfp4_grouped_bandwidth/n512.c').read_text().replace('"../../csrc/tpc/mxfp4_linear/gemv.c"','"'+str(root/'csrc/tpc/mxfp4_linear/gemv.c')+'"')
 (out/'grouped.c').write_text(source)
commands=[]
(out/'ring_gate.c').write_bytes((root/'benchmarks/mxfp4_grouped_bandwidth/ring_gate.c').read_bytes())
for name in ['grouped','decode','reduce','ring_gate']:
 if name=='grouped'and a.literal:
  ref=root/'benchmarks/mxfp4_grouped_bandwidth/reference';data=(ref/'gp.o').read_bytes()
  assert hashlib.sha256(data).hexdigest()=='fe7cd09c6bb2d20b02a6addc901916a44cc30a9cd03be3e9704116048ec72460'
  assembly,plan=transform((ref/'gp.s').read_text(),a.literal!='control',a.literal=='safe')
  (out/'grouped.s').write_text(assembly);(out/'grouped.o').write_bytes(data)
  (out/'grouped.c').write_text('// Executed code comes from the pinned reference ELF and grouped.s; no C compilation.\n')
  (out/'schedule.json').write_text(json.dumps(plan,indent=2)+'\n')
  commands += [['tpc-clang','-mcpu=gaudi2','-c','grouped.s','-o','grouped_slots.o'],['objcopy','--dump-section','.text=grouped.text','grouped_slots.o'],['objcopy','--update-section','.text=grouped.text','grouped.o'],['python3',str(root/'benchmarks/mxfp4_grouped_bandwidth/fix_literal_symbols.py'),'grouped.o','grouped_slots.o'],['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','grouped.o','grouped_x86.o'],['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','grouped.o']]
  continue
 if name=='reduce'and a.literal:
  (out/'reduce.c').write_text('''#define main unused_legacy_main
#include "'''+str(root/'csrc/tpc/mxfp4_linear/gemv.c')+'''"
#undef main
void main(tensor input,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int splits=get_dim_size(input,2);
 for(int e=b[2];e<end[2];++e)for(int row=b[1];row<end[1];++row)for(int n=b[0];n<end[0];++n){
  float128 total={0};for(int s=0;s<splits;++s){int5 p={n*128,row,s,e,0};total.v1+=v_f32_ld_tnsr_b(p,input);p[0]+=64;total.v2+=v_f32_ld_tnsr_b(p,input);}
  float128 y=linear_acc(total);int5 p={n*128,row,e,0,0};v_f32_st_tnsr(p,output,y.v1);p[0]+=64;v_f32_st_tnsr(p,output,y.v2);
 }
}
''')
 defines=([f'-DGK_SMALLM_ROWS={a.rows}','-DGK_SMALLM_UNROLL=4','-DGK_SMALLM_VECTOR=1','-DGK_SMALLM_FOLD_SCALE=1']if name=='grouped'else['-DGK_MXFP4_DECODE_UNROLL=4','-DGK_MXFP4_DECODE_FAST_SCALE=2']if name=='decode'else[])
 if name=='grouped'and a.window:defines += [f'-DGK_SMALLM_STRAIGHT={a.window}']
 if name=='grouped'and a.n512_window:defines += [f'-DGK_N512_WINDOW={a.n512_window}']
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*defines,flag,name+'.c','-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
 commands.append(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o'])
tile=512 if a.n512_window or a.literal else 256;literal=['-DGK_LITERAL_GP=1']if a.literal else[]
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,f'-DGK_SMALLM_ROWS={a.rows}',f'-DGK_GROUPED_TILE={tile}',*literal,str(root/'benchmarks/mxfp4_grouped_bandwidth/glue.cpp'),'grouped_x86.o','decode_x86.o','reduce_x86.o','ring_gate_x86.o','-o','libgrouped_tpc.so'])
commands.append(['g++','-O3','-std=c++17','-I'+a.include_dir,f'-DGK_GROUPED_TILE={tile}',*literal,str(root/'benchmarks/mxfp4_grouped_bandwidth/probe.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe'])
meta={'rows':a.rows,'window':a.window,'n512_window':a.n512_window,'literal':a.literal,'activation':a.activation,'lookup_x2':a.lookup_x2,'decode_canonical_zero':a.decode_canonical_zero,'decode_window':a.decode_window,'commands':commands,'state':'building','device_verified':False}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with(out/'build.log').open('w')as log:
  for command in commands:
   log.write(json.dumps(command)+'\n');log.flush()
   if command[0]=='tpc-llvm-objdump':
    with(out/(command[-1]+'.dis')).open('w')as dis:subprocess.run(command,cwd=out,stdout=dis,stderr=log,check=True)
   else:subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 meta['state']='built_not_device_verified'
finally:
 meta['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()and p.name!='build.json'}
 (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'state':meta['state'],'output':str(out)}))

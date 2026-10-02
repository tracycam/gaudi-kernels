"""Read sealed identities/trace and exhaustively check compact top8 address math."""
import argparse,collections,hashlib,io,json,re,subprocess
from pathlib import Path
from elftools.elf.elffile import ELFFile
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=a.canonical/'artifacts/builds/production-integration/model-runs/production-fp32-70-d/source'
def extract(path,symbol,name):
 raw=path.read_bytes();e=ELFFile(io.BytesIO(raw));syms={s.name:s for s in e.get_section_by_name('.symtab').iter_symbols()};start=syms['_binary_'+symbol+'_o_start'];end=syms['_binary_'+symbol+'_o_end'];section=e.get_section(start['st_shndx']);offset=section['sh_offset']+start['st_value']-section['sh_addr'];elf=raw[offset:offset+end['st_value']-start['st_value']];target=out/(name+'.elf');target.write_bytes(elf);text=ELFFile(io.BytesIO(elf)).get_section_by_name('.text').data();(out/(name+'.text')).write_bytes(text)
 cmd=['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(target)];dis=subprocess.check_output(cmd,text=True);(out/(name+'.dis')).write_text(dis)
 return {'source':str(path.relative_to(a.canonical)),'library_sha256':hashlib.sha256(raw).hexdigest(),'elf_sha256':hashlib.sha256(elf).hexdigest(),'text_sha256':hashlib.sha256(text).hexdigest(),'objdump_command':cmd},dis
assets={}
assets['post'],post=extract(source/'production-runtime/libraries/libgaudi_qkv_postprocess_tpc.so','cache_bf16_v2','post')
assets['gp'],gp=extract(source/'production-runtime/gp-scale-tail/tpc.so','candidate','gp')
assets['down'],down=extract(source/'production-runtime/libraries/libgaudi_down_activation_tpc.so','direct_down','down')
for dis,shift in [(gp,1),(down,3)]:
 assert '0xaaaaaaab' in dis and re.search(r'shr\.u32\s+S11, S11, (?:0x)?'+str(shift)+r'\b',dis)
 assert re.search(r'mul\.u32 upper32\s+S13, S11, S31',dis)
checks=[]
for m in (1,2,8,16,32,128,512,513):
 slots=m*8;recip=(1<<32)//8;counts={};maxima={}
 for kind,split,shift,ak in [('gp',3,1,6144),('down',12,3,256)]:
  maximum=-1
  for task in range(slots*split):
   route=((task*0xaaaaaaab)>>32)>>shift;sub=task-route*split
   token=(route*recip)>>32;rem=route-token*8
   if rem>=8:token+=1
   slot=route-token*8
   assert (route,sub,token,slot)==(task//split,task%split,(task//split)//8,(task//split)%8)
   assert 0<=token<m and 0<=slot<8
   first=token*6144+sub*2048 if kind=='gp' else route*256
   last=first+(2048 if kind=='gp' else 256)-1
   assert last<(m*6144 if kind=='gp' else slots*256)
   maximum=max(maximum,last)
   assert task*512+511<slots*(1536 if kind=='gp' else 6144)<2**31
  counts[kind]=slots*split;maxima[kind]=maximum
 checks.append({'M':m,'topk':8,'tasks_checked':counts,'max_activation_element_index':maxima,
  'gp_output_fp32_elements':slots*1536,'down_output_fp32_elements':slots*6144,'all_int32_bounds_pass':True})
model_path=source.parent/'result.json';model=json.loads(model_path.read_text());inv=model['policy_changes'][0][0]['qkv_postprocess']['prepared'];rejected=inv['rejected'];assert len(rejected)==10 and set(rejected.values())=={'attention_sliding_window'}
trace_path=a.canonical/'artifacts/builds/fp32-contract/trace-70d-boundaries-20260928/canonical-recheck/comparison-b/comparison.json';trace=json.loads(trace_path.read_text())['new'];recipes=[r for r in trace['recipes'] if any(n['op']=='rope_st2_fwd_bf16' for n in r['nodes'])];rope=[n for r in recipes for n in r['nodes'] if n['op']=='rope_st2_fwd_bf16'];assert len(recipes)==10 and len(rope)==20
report={'status':'PASS_OFFLINE_STATIC_AUDIT','device_qualified':False,'assets':assets,'compact_address_checks':checks,
 'compact_scope':'same top8 original owners and unchanged qualified ELF; no M>8 device gate. Index arithmetic only, not compiler slicing/placement or physical traffic.',
 'full_post':{'original_model_status':model['status'],'full_modules_only_window_rejected':rejected,'actual_rope_nodes_last_token':len(rope),'actual_rope_envelope_sum_us':sum(n['duration_us'] for n in rope),'full_compute_recipe_indices':[r['launch_in_token'] for r in recipes],
  'model_result_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'trace_comparison_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
  'scope':'rope-only observed work. Excludes DMA and unidentified fused V/gather; replacement cost and downstream schedule prevent treating this as saved time.'}}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'compact_shapes':len(checks),'full_rope_us':report['full_post']['actual_rope_envelope_sum_us']}))

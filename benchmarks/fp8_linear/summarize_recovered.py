"""Audit every captured graph and summarize same-input device comparisons."""
import argparse
import collections
import json
import re
from pathlib import Path
import statistics
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[];audits=[];pairs=[]
for grid in ['main','edges','stream']:
 case=a.root/('linear-'+grid+'-b');summary=json.loads((case/'summary.json').read_text());graphs=json.loads((case/'post_graph.json').read_text())['graphs']
 assert summary['returncode']==0 and len(graphs)==len(summary['results'])
 for rec,graph in zip(summary['results'],graphs):
  assert rec['passed'] and rec['changed_bf16_outputs_vs_baseline']==0
  tensors={t['name']:t for t in graph['tensors']};nodes=graph['nodes']
  assert tensors['activation']['max_shape']==[rec['K'],rec['M']]
  assert tensors['prepared_weight']['max_shape']==[rec['K'],rec['N']]
  assert tensors['prepared_weight']['dtype_bit_size']==8 and tensors['prepared_weight']['allocation']=='DRAM'
  intermediate_names=['fp8_linear_native_activation','fp8_linear_activation_scales','fp8_linear_unscaled_accumulation']
  intermediate=[tensors[name] for name in intermediate_names]
  assert all(t['allocation']=='SRAM' and not t['persistent'] for t in intermediate)
  mme=[n for n in nodes if n['engine']=='MME'];assert len(mme)==1
  assert mme[0]['input_tensors']==['fp8_linear_native_activation','prepared_weight']
  weight_readers=[n for n in nodes if 'prepared_weight' in n['input_tensors']];assert len(weight_readers)==1
  quant=nodes[0]['guid'];expected={'baseline':'fp8_linear_activation_fast','lut_single':'fp8_fp_quant_lut_single','lut_single4':'fp8_fp_quant_lut_single4'}[rec['variant']];assert quant==expected
  assert len(nodes)==3 and [n['engine'] for n in nodes]==['TPC','MME','TPC']
  assert tensors['fp8_linear_native_activation']['dtype_bit_size']==8 and tensors['fp8_linear_unscaled_accumulation']['dtype']=='float32'
  rec=dict(rec,grid=grid);records.append(rec)
  audits.append({'grid':grid,'M':rec['M'],'N':rec['N'],'K':rec['K'],'variant':rec['variant'],
   'graph_id':graph['id'],'engines':[n['engine'] for n in nodes],'quant_guid':quant,
   'intermediate_tensors':[{'name':t['name'],'shape':t['max_shape'],'dtype':t['dtype'],'allocation':t['allocation'],'persistent':t['persistent']} for t in intermediate],
   'logical_weight_tensor_bytes':rec['N']*rec['K'],'logical_full_weight_graph_consumers':len(weight_readers),
   'physical_weight_bus_bytes':None,'mme_strategy':mme[0].get('mme_node_strategy'),'mme_compute_utilization':mme[0].get('mme_compute_utilization'),
   'pass':True})
groups=collections.defaultdict(dict)
for r in records:groups[(r['grid'],r['M'],r['N'],r['K'],r['kind'])][r['variant']]=r
for key,group in groups.items():
 baseline=group['baseline']
 for variant in ['lut_single','lut_single4']:
  candidate=group[variant];assert candidate['fixture_sha256']==baseline['fixture_sha256']
  pairs.append({'grid':key[0],'M':key[1],'N':key[2],'K':key[3],'kind':key[4],'variant':variant,
   'baseline_event_us':baseline['event_median_us'],'candidate_event_us':candidate['event_median_us'],'candidate_wall_us':candidate['wall_median_us'],
   'latency_reduction_percent':100*(1-candidate['event_median_us']/baseline['event_median_us']),
   'speedup':baseline['event_median_us']/candidate['event_median_us'],'effective_tflops':candidate['effective_tflops'],
   'logical_weight_payload_gb_s':candidate['logical_weight_payload_gb_s'],'changed_bf16_outputs':candidate['changed_bf16_outputs_vs_baseline']})
quant=json.loads((a.root/'quant-byte-suite-b/summary.json').read_text());quant_results=[]
for r in quant:
 check=next(row for row in r['rows'] if row['stage']=='correctness');assert check['code_mismatches']==check['scale_word_mismatches']==0
 times=[row for row in r['rows'] if row['stage']=='timing'];assert len(times)==5
 quant_results.append({'case':r['case'],'variant':r['variant'],'M':r['M'],'K':r['K'],'checked_bytes':check['checked_bytes'],
  'code_mismatches':0,'scale_word_mismatches':0,'event_median_us':statistics.median(t['event_us'] for t in times),'wall_median_us':statistics.median(t['wall_us'] for t in times)})
profile=json.loads((a.root/'lut4-timeline.json').read_text())['records'][0]
node_durations=collections.defaultdict(list)
for invocation in profile['invocations']:
 for node in invocation['nodes']:node_durations[re.sub(r' iter \d+$','',node['name'])].append(node['end_us']-node['start_us'])
launch=json.loads((a.root/'linear-main-b/launch.json').read_text())
output={'device':{'module':launch['module'],'index':launch['index'],'PCI':launch['bus_id'],'boot_id':launch['preflight']['process']['boot_id']},'source_commit':launch['git_commit'],
 'unprofiled_recipe_count':len(records),'complete_bf16_outputs_checked':sum(r['M']*r['N'] for r in records),
 'quantization_byte_checks':sum(r['checked_bytes'] for r in quant_results),'pairs':pairs,'quantization':quant_results,'placement':audits,
 'maximum_wall_over_event_ratio':max(r['wall_median_us']/r['event_median_us'] for r in records),
 'max_relative_fp64':max(r['full_output_metrics']['reference_fp64']['relative_l2'] for r in records),
 'max_relative_gpu_style_fp64':max(r['full_output_metrics']['gpu_style_w8a8_fp64']['relative_l2'] for r in records),
 'profile_node_median_us':{k:statistics.median(v) for k,v in node_durations.items()},'profile_envelope_medians':profile['median'],
 'scope':'Real single-device native recipes; complete-output MAC oracle and bitwise baseline comparison. No model TPS, physical BMON byte count, or automatic dispatch claim.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ['pairs','quantization','placement']},indent=2))

"""Verify immutable raw assets, final output bytes and compiler physical graphs."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--canonical-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();archive=a.archive.resolve();remote=archive/'remote';base=archive.relative_to(a.canonical_root.resolve());out=a.output;out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((remote/'asset-manifest.json').read_text())
for name,info in manifest['files'].items():
 path=remote/name;assert path.stat().st_size==info['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],name
for name in ['norm-native-final-per-row-m1','norm-native-final-separate-m1']:
 for actual,expected in [('native_a8.bin','expected_native.bin'),('native_scale.bin','expected_scale.bin'),('residual_out.bin','residual_out.bin')]:
  assert (remote/'results'/name/actual).read_bytes()==(remote/'results/norm-native-a/1-6144/per_row'/expected).read_bytes(),(name,actual)
def load(name):return json.loads((remote/'results'/name/'result.json').read_text())
for name in ['norm-full-c','norm-quant-a','norm-quant-unroll4','norm-native-a','norm-quant-final','norm-production-qkv']:assert load(name)['status']=='PASS'
physical=[]
graphs=json.loads((remote/'results/norm-production-qkv/post_graph.json').read_text())['graphs']
for i,(m,fused) in zip([1,2,4,5,7,8],[(1,False),(1,True),(64,False),(64,True),(513,False),(513,True)]):
 g=graphs[i];nodes=[n for n in g['nodes'] if not n.get('is_logical')];norms=[t for t in g['tensors'] if t['dtype']=='bf16' and not t.get('persistent')]
 physical.append({'M':m,'fused':fused,'graph_name':g['name'],'physical_node_count':len(nodes),'nodes':[{'guid':n['guid'],'engine':n.get('engine')} for n in nodes],'nonpersistent_bf16':[{'name':t['name'],'shape':t['max_shape'],'allocation':t['allocation']} for t in norms]})
assert [v['physical_node_count'] for v in physical]==[4,3,4,3,11,10]
native_final={}
for name in ['norm-native-final-separate-m1','norm-native-final-per-row-m1']:
 lines=(remote/'results'/name/'run.log').read_text().splitlines();samples=[json.loads(l) for l in lines if l.startswith('{') and 'event_us' in l]
 native_final[name]={'samples':samples,'median_event_us':statistics.median(s['event_us'] for s in samples),'median_wall_us':statistics.median(s['wall_us'] for s in samples),'bytes_match_saved_cpu_reference':True}
results={'scope':'device-tested operators and explicit graph composition; not production/model TPS acceptance','archive':str(base),'remote_files_verified':manifest['count'],'remote_bytes_verified':manifest['bytes'],'native_final':native_final,'production_qkv':load('norm-production-qkv'),'production_physical_graphs':physical,'bf16':load('norm-full-c'),'quant_final':load('norm-quant-final'),'native_candidate_grid':load('norm-native-a'),'failures_retained':['builds/torch-a/build.json','results/norm-quick-a','results/norm-quick-b','builds/tpc-d/build.log'],'device_released':'results/norm-native-final-separate-m1/device-after.txt'}
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
files={str(path.relative_to(a.canonical_root.resolve())):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(archive.rglob('*')) if path.is_file()}
(out/'assets.json').write_text(json.dumps({'files':files,'count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'remote_files_verified':manifest['count'],'remote_bytes_verified':manifest['bytes']},indent=2)+'\n');print(json.dumps({'canonical_files_verified':len(files),'canonical_bytes':sum(v['bytes'] for v in files.values()),'remote_files_verified':manifest['count']}))

"""PostGraph placement/logical-read audit for the unexposed whole-chain graphs."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'benchmarks/block_fp8_framework'))
from audit import audit
report=json.loads((a.directory/'summary.json').read_text());graphs=json.loads((a.directory/'post_graph.json').read_text())['graphs'];rows=[]
for graph in graphs:
 if not any(n['engine']=='MME' for n in graph['nodes']):continue
 reducers=[n for n in graph['nodes'] if n['guid'].startswith(('gk_block128_reduce_neumaier_row1_v1','gk_neumaier_isa_handschedule_v1'))]
 assert reducers,graph['name']
 placement=audit(graph);tensors={t['name']:t for t in graph['tensors']};partial=[]
 for node in graph['nodes']:
  if node['engine']=='MME':
   for name in node['output_tensors']:
    t=tensors[name];partial.append(dict(name=name,dtype_bits=t['dtype_bit_size'],allocation=t['allocation'],persistent=t['persistent']))
 rows.append(dict(placement=placement,reducers=sorted(set(n['guid'] for n in reducers)),mme_outputs=partial,mme_outputs_sram=bool(partial) and all(t['allocation']=='SRAM' and not t['persistent'] for t in partial)))
expected=len({r['case'] for r in report['numerics'] if r['phase']=='chain'})*2
complete=len(rows)==expected and expected>0
passed=complete and report.get('all_pass',False) and all(r['mme_outputs_sram'] and r['placement']['no_logical_weight_read_amplification'] for r in rows)
result=dict(all_pass=passed,expected_chain_graphs=expected,found_chain_graphs=len(rows),records=rows,scope='PostGraph storage and logical views, not physical HBM counters; launch counts are separately observed by the replay API interposer')
(a.directory/'placement-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if passed else 1)

"""CPU-only adversarial checks for previously identified acceptance loopholes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import PreparedSeparableFP8
from gaudi_kernels.mxfp4 import PreparedMXFP4
from audit_w8a16 import audit
from validation import timings_agree

for value in [PreparedSeparableFP8(None,None,None,99),PreparedMXFP4(None,None,1,1,99)]:
    try:value.to('cpu')
    except ValueError:pass
    else:raise AssertionError('unknown version accepted')
assert timings_agree([20,22],[21,23])
for events,walls in [([],[]),([0],[20]),([float('nan')],[20]),([20],[float('inf')]),([1],[50]),([50],[1]),([1,2],[1])]:
    assert not timings_agree(events,walls)

base=root/'evidence/framework/results/graph-d-w8a16-placement'
data=json.loads((base/'post_graph.json').read_text())
assert audit(base/'post_graph.json')['status']=='PASS'
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory);(directory/'result.json').write_bytes((base/'result.json').read_bytes())
    graph_path=directory/'post_graph.json'
    broken=copy.deepcopy(data);graph=broken['graphs'][0]
    decode=next(n for n in graph['nodes'] if n['guid']=='fp8_linear_decode')
    name=decode['output_tensors'][0]
    tensor=copy.deepcopy(next(t for t in graph['tensors'] if t['name']==name))
    tensor.update(name='hidden_expanded_hbm_copy',allocation='DRAM',persistent=True)
    graph['tensors'].append(tensor)
    for node in graph['nodes']:
        if node['guid']=='gemm':node['input_tensors']=[tensor['name'] if t==name else t for t in node['input_tensors']]
    graph['nodes'].append({'name':'bad_copy','guid':'memcpy','input_tensors':[name],'output_tensors':[tensor['name']]})
    graph_path.write_text(json.dumps(broken));assert audit(graph_path)['status']=='FAIL'
    broken=copy.deepcopy(data);broken['graphs']=broken['graphs'][:1];graph_path.write_text(json.dumps(broken))
    try:audit(graph_path)
    except AssertionError:pass
    else:raise AssertionError('incomplete compiled fixture coverage accepted')
print(json.dumps({'status':'PASS','checks':['version before transfer','timing finite positive agreement','downstream HBM weight copy','missing compiled fixture']}))

"""Synthetic graph replay of the device page/slot producer; no model claim."""
import json
import os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1'
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from device_metadata import query_metadata

def sync():hc.mark_step();torch.hpu.synchronize()

out=Path(os.environ['PROBE_OUT']);records=[]
with torch.inference_mode():
    for capacity in (4,8):
        cpu_pages=torch.arange(120,dtype=torch.int32).reshape(3,40)+10
        pages=cpu_pages.to('hpu');lengths=torch.zeros(3,dtype=torch.int32,device='hpu')
        counts=torch.ones(3,dtype=torch.int32,device='hpu');sync()
        graph=torch.hpu.HPUGraph()
        with torch.hpu.graph(graph):
            metadata=query_metadata(pages+0,lengths+0,counts+0,capacity,128,128,999,1000)
            # Keep a dependent consumer in the graph to check live producer edges.
            valid_count=(metadata['block_groups']>=0).sum()
        sync()
        for start in (0,1,125,126,127,128,129,4094,4095,5119):
            for query_counts in ((1,2,capacity),(capacity,capacity,capacity),(0,3,1)):
                cpu_lengths=torch.tensor([start,start+2,start+4],dtype=torch.int32)
                cpu_counts=torch.tensor(query_counts,dtype=torch.int32)
                expected=query_metadata(cpu_pages,cpu_lengths,cpu_counts,capacity,128,128,999,1000)
                lengths.copy_(cpu_lengths);counts.copy_(cpu_counts);sync();graph.replay();sync()
                actual={k:v.cpu()for k,v in metadata.items()}
                passed=all(torch.equal(actual[k],v)for k,v in expected.items())
                passed=passed and int(valid_count.cpu())==int((expected['block_groups']>=0).sum())
                records.append(dict(capacity=capacity,start=start,counts=query_counts,fault=actual['fault'].tolist(),passed=passed))
                assert passed,records[-1]
        del graph;sync()
(out/'result.json').write_text(json.dumps(dict(status='PASS_DEVICE_QUERY_METADATA',records=records,
    scope='Device page/slot producer and dependent consumer only; attention integration, native cycle and TPS still unqualified'),indent=2)+'\n')

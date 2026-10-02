"""Independent original planner + exclusive chunk prefixes."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('v1_reference',Path(__file__).resolve().parents[1]/'moe_route_metadata/reference.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
def reference(ids,experts,rows,capacity):
    result=v1.reference(ids,experts,rows,capacity);flat=[e for row in ids for e in row];q=(len(flat)+63)//64
    offsets=[]
    for e in range(experts):
        row=[0]
        for i in range(q):row.append(row[-1]+sum(x==e for x in flat[i*64:(i+1)*64]))
        offsets.append(row)
    result['chunk_offsets']=offsets
    return result

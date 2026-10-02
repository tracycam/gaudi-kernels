"""Independent integer reference; float0/1 implementation must match exactly."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('route_v1_reference',Path(__file__).resolve().parents[1]/'moe_route_metadata/reference.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)


def reference(ids,experts,rows,capacity):
    result=v1.reference(ids,experts,rows,capacity)
    flat=[x for row in ids for x in row]
    result['chunk_counts']=[[flat[q:q+64].count(e)for q in range(0,len(flat),64)]for e in range(experts)]
    assert [sum(row)for row in result['chunk_counts']]==result['counts']
    return result

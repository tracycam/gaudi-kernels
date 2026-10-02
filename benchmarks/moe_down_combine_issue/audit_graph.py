"""PostGraph owner/descriptor audit, with no HBM-traffic inference."""
import argparse,json,math
from pathlib import Path

def read_graphs(path):
    files=[path] if path.is_file() else sorted(path.rglob('*.json'))
    return [g for file in files for g in json.loads(file.read_text())['graphs']]

def audit(path,experts,variants):
    rows=[];allgraphs=read_graphs(path)
    targets={'baseline':'downact_direct_down_broadcast',**{v:'gk_down_combine_m1_'+v+'_v0' for v in variants}}
    for variant,guid in targets.items():
        gs=[g for g in allgraphs if any(n['guid']==guid for n in g['nodes']) and any(n['guid']=='gk_moe_gp_scale_tail_v1' for n in g['nodes'])]
        assert gs,'missing full-chain compiled graph '+variant
        for g in gs:
            ts={t['name']:t for t in g['tensors']};producer={name:n for n in g['nodes'] for name in n.get('output_tensors',[])}
            nodes=[n for n in g['nodes'] if not n.get('is_logical',False)]
            n=next(n for n in g['nodes'] if n['guid']==guid)
            assert n['tpc_working_engines']==[24],(variant,n.get('tpc_working_engines'))
            assert sum(x['guid']=='pf_combine_f32' for x in g['nodes'])==(1 if variant=='baseline' else 0)
            entries=[]
            for i,(fcd,height) in enumerate([(256,experts*3072),(512,experts*96)]):
                t=ts[n['input_tensors'][i]];expected=[fcd,height] if variant=='baseline' else [fcd//2,height*2]
                assert t['dtype']=='uint8' and t['max_shape']==expected,(variant,t['max_shape'],expected)
                current=t;ancestry=[]
                while current['name'] in producer:
                    p=producer[current['name']]
                    assert p.get('is_logical',False) and p['guid'] in ('reshape','static_reshape'),('weight copy/transform',p['guid'])
                    assert len(p['input_tensors'])==1
                    ancestry.append(p['guid']);current=ts[p['input_tensors'][0]]
                assert current['persistent'] and current['dtype']=='uint8'
                assert math.prod(current['max_shape'])==fcd*height
                if variant!='baseline':
                    assert ancestry or t.get('alias',False),('no compiled logical alias evidence',t['name'])
                entries.append(dict(compiled_fcd=expected[0],shape=expected,allocation=t['allocation'],root_shape=current['max_shape'],root_persistent=current['persistent'],logical_alias_path=ancestry,alias_flag=t.get('alias',False),root_user_section=current.get('user_mem_section_index'),root_user_offset=current.get('user_mem_offset')))
            rows.append(dict(variant=variant,id=g['id'],graph_name=g['name'],physical_nodes=len(nodes),guids=[x['guid'] for x in nodes],workspace_bytes=g.get('workspace_size'),weights=entries))
    return dict(pass_owner_and_descriptor_gate=True,rows=rows,physical_HBM_measured=False,scope='Valid descriptor elements and logical path to one original persistent owner; excludes hardware fetch over-read/cache traffic')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--postgraph',type=Path,required=True);p.add_argument('--experts',type=int,required=True);p.add_argument('--variants',nargs='+',default=['p6','p4']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.postgraph,a.experts,a.variants);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

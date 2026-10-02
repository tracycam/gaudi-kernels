"""Recompute committed evidence from local raw assets, without device imports."""
import argparse
import json
from pathlib import Path
import statistics as S


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def medians(group, fields):
    return {field: S.median(x[field] for x in group) for field in fields}


def summarize(root):
    report = dict(schema=1, scope='measured cases only; microbenchmarks are not model TPS',
                  exits=[], binding_controls=[], tp8={}, models=[])
    for p in sorted(root.glob('*/results/*/exit.json')):
        x=json.loads(p.read_text())
        report['exits'].append(dict(case=str(p.relative_to(root).parent),
            code=x.get('runner_exit_code',x.get('returncode')),reason=x.get('reason')))
    for p in sorted(root.glob('model-*/executor/*.exit.json')):
        x=json.loads(p.read_text());report['exits'].append(dict(case=str(p.relative_to(root)),
            code=x.get('returncode',x.get('exit_code')),reason=x.get('reason')))
    for p in sorted(root.glob('*/results/bindings-*/bindings.jsonl')):
        data=rows(p)
        for key in sorted({(x['bindings'],x['nodes'],x['width'])for x in data
                           if x['stage']=='binding_timing'and x.get('controlled_nodes')}):
            group=[x for x in data if x['stage']=='binding_timing'and x.get('controlled_nodes')
                   and (x['bindings'],x['nodes'],x['width'])==key]
            report['binding_controls'].append(dict(case=str(p.relative_to(root)),
                bindings=key[0],nodes=key[1],width=key[2],trials=len(group),
                **medians(group,['host_call_us','event_us','workspace_bytes'])))
        assert all(x['bad']==0 for x in data if x['stage']=='binding_gate'),str(p)
    for fixture in ['exact','ordinary']:
        files=sorted((root/'runtime-i/results/owned-tp8-i'/fixture).glob('rank*.jsonl'))
        if not files:continue
        assert len(files)==8
        ranks={variant:[]for variant in range(3)};quality=[]
        for p in files:
            data=rows(p);quality += [x for x in data if x['stage']=='quality']
            for variant in ranks:
                group=[x for x in data if x['stage']=='timing'and x['variant']==variant]
                fields=['event_us_per_chain','wall_us_per_chain','submit_us_per_chain']
                if variant!=2:fields += ['ag_host_us']
                ranks[variant].append(medians(group,fields))
        assert all(x['bad']==0 for x in quality)
        report['tp8'][fixture]=dict(gates=len(quality),checked=sum(x['checked']for x in quality),
            variants={str(k):medians(values,values[0].keys())for k,values in ranks.items()})
    for p in sorted(root.glob('model-*/executor/*/result.json')):
        x=json.loads(p.read_text());bench=x.get('replay_bench',[])
        if not bench or 'original'not in bench[0]:continue
        assert len(bench)==8 and all(q['status']=='BITWISE_PASS'for q in bench)
        record=dict(case=str(p.relative_to(root).parent),result_status=x['status'],
            scope=bench[0]['scope'],counts=[q['counts']for q in bench],arms={},ownership=[],owned_tail=[])
        for arm in ['original','normalized']:record['arms'][arm]=medians([q[arm]for q in bench],bench[0][arm].keys())
        for owner in sorted(p.parent.glob('owned-ownership-rank*.json')):
            q=json.loads(owner.read_text());record['ownership'].append(dict(rank=int(owner.stem.rsplit('rank',1)[1]),
                owners=len(q['tensor_owners']),unknown_reads=len(q['unknown_first_reads']),
                unknown_unique_ranges=len({(a.get('address'),a.get('bytes'))for a in q['unknown_first_reads']}),
                declared_reads=len(q['declared_first_reads']),sync_commands=q['sync_commands'],
                workspace_max_bytes=q.get('workspace_max_bytes')))
        for q in bench:
            if q.get('owned_tail')is not None:
                tail=q['owned_tail'];record['owned_tail'].append(dict(rank=q['rank'],status=tail['status'],
                    gates=len(tail['gates']),workspace_bytes=tail['report']['workspace_bytes'],
                    pool_bytes=tail['report']['pool_bytes'],scope=tail['scope']))
        report['models'].append(record)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=summarize(args.assets);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(exits=len(data['exits']),controlled_shapes=len(data['binding_controls']),
                         tp8_fixtures=len(data['tp8']),models=len(data['models']))))

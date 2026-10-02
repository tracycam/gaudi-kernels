"""Summarize complete-chain batch evidence without conflating timing scopes."""
import argparse
import json
from pathlib import Path
import statistics


def distribution(values):
    return dict(count=len(values),median_ms=statistics.median(values)/1e6 if values else None,
                minimum_ms=min(values)/1e6 if values else None,
                maximum_ms=max(values)/1e6 if values else None)


def analyze(root):
    result=json.loads((root/'result.json').read_text())
    report=dict(status=result['status'],layers=result['layers'],candidate_accepted=False,
        scope='execution equivalence and bounded teacher quality; actual arrivals and rank timings separate',
        runs=[],teacher=[])
    for row in result['runs']:
        client=row.get('token_arrivals')
        if client is not None:
            from .token_arrivals import summarize
            client=dict(client,summary=summarize(client['events'],client['request_ids'],
                warmup_tokens=client['summary']['warmup_tokens']))
        ranks=[]
        for rank in row['ranks']:
            steps=[s for s in rank['native_steps'] if s.get('kind')=='batch']
            fields=('native_wall_ns','worker_wall_ns','cpu_metadata_ns','cpu_commit_ns',
                    'enqueue_ns','synlaunch_ns','hccl_ns')
            # Compact mode deliberately leaves per-API counters at zero. Old
            # batch records omitted the validity flag: do not infer free SDK
            # launch/communication from those zeros either.
            ranks.append(dict(rank=rank['rank'],timings={
                field:distribution([s[field] for s in steps if field in s and
                    (field not in ('synlaunch_ns','hccl_ns') or s.get('api_detail_valid',False))])
                for field in fields},
                batch_step_count=len(steps),
                submission_counts=sorted({c['submission_count'] for c in rank['captures']
                    if c.get('kind')=='batch' and 'submission_count' in c}),
                capture_count=len(rank['captures'])))
        report['runs'].append(dict(batch=row['batch'],arm=row['arm'],active=row['active'],
            policy=row.get('policy'),comparison=row.get('comparison','bridge_native'),
            matches_bridge=row['matches_bridge'],ranks=ranks,
            engine_client=client,
            generate_wall_ms=row['wall_ns']/1e6,
            wall_scope='includes admission, prefill, first token, recapture and completion'))
    for path in sorted(root.glob('b*-teacher.json')):
        doc=json.loads(path.read_text());checks=[c['check'] for c in doc['checks']]
        report['teacher'].append(dict(batch=doc['batch'],passed=doc['pass'],queries=len(checks),
            per_request_queries={str(i):sum(c['request']==i for c in doc['checks']) for i in range(doc['batch'])},
            max_kl=max(c['max_kl_ref_to_candidate'] for c in checks),
            max_relative_l2=max(c['relative_l2'] for c in checks),
            bitwise_equal_queries=sum(c['bitwise_equal'] for c in checks),
            scope=doc['scope']))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.root),indent=2)+'\n')

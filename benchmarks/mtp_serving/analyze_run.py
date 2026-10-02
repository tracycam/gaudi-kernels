"""Summarize measured bridge acceptance/routes without inventing native TPS."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from metrics import acceptance,routes


def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def summarize(root):
    result=json.loads((root/'result.json').read_text())
    report={k:result.get(k)for k in ('status','mode','drafter','corpus','ignore_eos','quality_status')}
    if not result['config'].get('speculative_config'):
        report['drafter']='none (from recorded engine config)'
    report.update(candidate_accepted=False,performance_qualified=False,runs=[])
    hashes={}
    for run in result['runs']:
        b,phase=run['B'],run['phase']
        tag=run.get('diagnostic_tag',f'b{b}-{phase}')
        raw=json.loads((root/f'{tag}-mtp.json').read_text())
        identity=[digest(o['ids'])for o in run['outputs']]
        variant=run.get('variant','baseline')
        block=run.get('block',0)
        record=dict(B=b,phase=phase,variant=variant,block=block,output_tokens=[len(o['ids'])for o in run['outputs']],output_sha256=identity,
                    retrieval_pass=run.get('retrieval_pass'),mature=run.get('mature'),
                    worker_acceptance=acceptance(raw['steps']))
        hashes.setdefault(f'{variant}/block{block}/b{b}',{})[phase]=identity
        audit=raw.get('audit',[])
        if audit:
            rows=[x['checks']['same_prefix_row']for x in audit if 'same_prefix_row'in x['checks']]
            record['same_input_audit']=dict(cases=len(audit),restored_full_replay_bitwise=all(
                all(v['bitwise_equal']for k,v in x['checks'].items()if k!='same_prefix_row')for x in audit))
            if rows:
                record['same_input_audit'].update(row_cases=len(rows),max_kl=max(v['kl_batched_to_single']for v in rows),
                    max_abs=max(v['max_abs']for v in rows),argmax_changed=sum(v['argmax_changed']for v in rows),
                    all_row_logits_bitwise=all(v['bitwise_equal']for v in rows))
        # Terminal worker outputs can exceed the scheduler's output budget.
        # Report unclipped proposal statistics, separately from emitted TPS.
        by_request=defaultdict(list)
        for step in raw['steps']:
            for row in step['requests']:by_request[row['request_id']].append(row)
        record['acceptance_excluding_terminal_cycle']=acceptance([
            dict(requests=rows[:-1])for rows in by_request.values()])
        grouped={}
        expected_layers=run['controls'][0]['target_moe_layers']
        for frame in raw['frames']:
            counts=tuple(n for _,n in frame['requests']);key=','.join(map(str,counts))
            group=grouped.setdefault(key,dict(T=list(counts),frames=0,layer_samples=0,U=[],routes=0,unique=0,m_histogram=Counter()))
            group['frames']+=1
            for layer in routes(frame,expected_layers):
                group['layer_samples']+=1;group['U'].append(layer['U']);group['routes']+=layer['routes'];group['unique']+=layer['U']
                group['m_histogram'].update(layer['m_histogram'])
        for group in grouped.values():
            values=group.pop('U');group['unique_experts']=dict(min=min(values),mean=statistics.mean(values),max=max(values))
            group['maximum_logical_weight_saving']=1-group['unique']/group['routes']
            group['m_histogram']=dict(group['m_histogram'])
        record['route_groups']=list(grouped.values());report['runs'].append(record)
    report['phase_token_identity']={b:dict(hashes=phases,all_equal=len({tuple(v)for v in phases.values()})==1)for b,phases in hashes.items()}
    report['scope']='Recorded installed first-head-recurrent bridge baseline. Route readback is untimed; acceptance is not output quality. No native/MTP3/DSpark/DFlash or target-only equivalence claim.'
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=summarize(a.case);a.output.write_text(json.dumps(r,indent=2)+'\n')
    for v in r['runs']:
        print(json.dumps({k:v[k]for k in('B','phase','output_tokens','mature','acceptance_excluding_terminal_cycle')}))


if __name__=='__main__':main()

"""Report completed phases of a failed/full FP32 run without upgrading status."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('case',type=Path)
    p.add_argument('--executor',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.executor/'executor'))
    from fp32_quality_contract import classify
    source=a.case/'result.json';d=json.loads(source.read_text())
    assert d['quality_contract']=='fp32_arithmetic_v1'
    quality=classify(d,d['quality_plan'],(0,16,48),70)
    rows=[r for r in d['teacher_forced']['rows'] if r['quality_role']=='candidate']
    records=[r for row in rows for w in row['same_input_qkv_audit'] for r in w['same_input_qkv_audit']]
    assert len(records)==1680 and all(r['fp32_contract']['passed'] for r in records)
    assert all(r['plain_vs_audit_logits']['all_bits_equal'] for r in rows)
    arms=[r for r in d['runs'] if 'native-policy-abba-' in r['name']]
    assert len(arms)==4 and all(r['token_count_pass'] and r['bitwise_token_match'] for r in arms)
    values=[r.get('serving_timing',r.get('diagnostic_timing')) for r in arms]
    def pooled(indices):
        tokens=sum(values[i]['steady_steps'] for i in indices)
        seconds=sum(values[i]['steady_steps']/values[i]['tps'] for i in indices)
        return dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,ms_per_token=1000*seconds/tokens)
    control,candidate=pooled((0,3)),pooled((1,2))
    result=dict(case=a.case.name,source_status=d['status'],source_candidate_accepted=d['candidate_accepted'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        complete_quality_components=quality,staged_records=len(records),
        zero_incompatible_partial_cells=all(r['fp32_contract']['stages']['partial']['incompatible_cells']==0 for r in records),
        all_orders_exact_partial_cells=sum(r['fp32_contract']['stages']['partial']['all_orders_exact_cells'] for r in records),
        all_plain_staged_logits_equal=True,
        required_references=d['same_contract_reference'],adaptation=d['native_half_adaptation'],
        abba=dict(arms=[v['tps'] for v in values],control=control,candidate=candidate,
                  saved_ms_per_token=control['ms_per_token']-candidate['ms_per_token'],
                  remaining_ms_to_100=candidate['ms_per_token']-10,
                  interpretation='no resolved B1 gain; identical warmup and all stalls retained; model acceptance incomplete'),
        chat_completed=len(d.get('chat_quality',[])),batch_completed=len(d.get('batch_runs',[])),
        long_context_completed=d.get('long_context',{}).get('status'),error=d.get('error'),
        scope='completed numerical/performance phases only; original FAIL and missing acceptance phases remain unchanged')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    assert not a.out.exists()
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source_status','staged_records','abba','chat_completed','batch_completed')}))


if __name__=='__main__':main()

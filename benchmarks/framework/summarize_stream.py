"""Supplementary module-4 FP8 measurements; keep them separate from module 5."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('fp8_audit',Path(__file__).resolve().parents[1]/'fp8_linear/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
baseline=a.root/'fp8-stream-baseline-module4';candidate=a.root/'fp8-stream-rows-module4'
records=[]
for r in json.loads((candidate/'summary.json').read_text()):
    label=f"{r['mode']}-{r['M']}-{r['N']}-{r['K']}-{r['kind']}"
    old=json.loads((baseline/label/'result.json').read_text())
    fixture=f"fixture-{r['M']}-{r['N']}-{r['K']}-{r['kind']}"
    hashes=json.loads((candidate/fixture/'fixture.json').read_text())['sha256']
    assert hashes==json.loads((baseline/fixture/'fixture.json').read_text())['sha256']
    for directory in [baseline,candidate]:
        for name,sha in hashes.items():
            assert hashlib.sha256((directory/fixture/name).read_bytes()).hexdigest()==sha
    actual=(candidate/label/'output.bin').read_bytes()
    assert len(actual)==r['M']*r['N']*2 and actual==(baseline/label/'output.bin').read_bytes()
    for row in [old,r]:
        assert row['returncode']==0 and next(v for v in row['rows'] if v['stage']=='correctness')['pass']
    placement=audit.audit(next((candidate/label/'graphs').glob('*PostGraph-symbol.pbtxt')),r)
    assert placement['pass'],placement
    records.append({'label':label,'M':r['M'],'N':r['N'],'K':r['K'],'mode':r['mode'],
        'baseline_us':old['event_median_us'],'candidate_us':r['event_median_us'],
        'effective_TBps':r['weight_payload_gb_s']/1000,
        'weight_bandwidth_percent_of_2_45_TBps':r['weight_payload_gb_s']/2450*100,
        'effective_TFLOPS':r['effective_tflops'],
        'compute_percent_of_nominal_peak':r['effective_tflops']/(865 if r['mode']=='w8a8' else 432)*100,
        'outputs':r['M']*r['N'],'bitwise_equal':True,'placement':placement})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'status':'PASS','module':4,'records':records,
    'scope':'native same-card supplementary probes; effective ratios, not physical bus/issue counters'},indent=2)+'\n')
for r in records:print(json.dumps({k:v for k,v in r.items() if k!='placement'}))

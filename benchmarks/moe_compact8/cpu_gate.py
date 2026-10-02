"""Small integer/interval tests only: do not allocate/read expert weights."""
import argparse
import json
from pathlib import Path
from ledger import routes,accounting
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for tokens in (2,8):
    for state in ('sparse','hot','rotated','zero'):
        r=accounting(routes(tokens,state));r.pop('intervals');r['state']=state
        assert r['resident_weight_scale_bytes']==962592768
        assert r['requested_weight_scale_bytes_each_mode']==tokens*8*2506752
        if state=='hot':assert r['unique_selected_weight_scale_bytes']==20054016
        else:assert r['unique_selected_weight_scale_bytes']==tokens*8*2506752
        rows.append(r)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'PASS_INTEGER_LEDGER',
    'large_weights_constructed':False,'device_verified':False,'records':rows},indent=2)+'\n')
print('PASS_INTEGER_LEDGER',len(rows))

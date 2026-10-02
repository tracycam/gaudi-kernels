"""Index completed overhead probes, retaining original records unmodified."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('builds',type=Path);a=p.parse_args()
summary={'scope':'complete synthetic operator/chain recipes, not model throughput or physical bus counters','probes':[]}
for path in sorted(a.evidence.glob('bf16-*/result.json')):
 data=json.loads(path.read_text());records=data.get('records',[])
 numerical=[r for r in records if 'correctness' in r]
 if not numerical:continue
 rows=[]
 for r in numerical:
  depth=r.get('depth',1);time=r['timing']['median_event_us']
  rows.append({**r,'analysis':{'linear_nodes_per_recipe':depth,'event_us_per_linear':time/depth,'total_useful_TFLOPS':2*r['m']*r['n']*r['k']*depth/time/1e6}})
 summary['probes'].append({'case':path.parent.name,'source_identity_verified':json.loads((path.parent/'launch.json').read_text())['source_identity_verified'],'configurations':len(rows),'full_output_checks':sum(r['correctness']['checked'] for r in rows),'failed':[r['label'] for r in rows if r['returncode'] or not r['correctness']['screen_pass']],'records':rows})
(a.evidence/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files={}
for base in [a.evidence,a.builds]:
 for path in sorted(base.rglob('*')):
  if path.is_file() and path.name not in {'SHA256SUMS.json','.gitignore'}:files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(a.evidence/'SHA256SUMS.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'probes':[(x['case'],x['configurations'],len(x['failed'])) for x in summary['probes']],'files':len(files)}))

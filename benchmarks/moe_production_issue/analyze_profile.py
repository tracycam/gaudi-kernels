"""Use only the final verified ABBA timing replay phase from the physical trace."""
import argparse,collections,csv,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=list(csv.DictReader(a.csv.open()));names={'gk_moe_gp_folded_v1':'control','gk_moe_gp_scale_tail_v1':'candidate'}
gp=sorted([r for r in rows if r['Op Type'] in names],key=lambda r:float(r['Start time of node']))
assert collections.Counter(names[r['Op Type']] for r in gp)=={'control':141,'candidate':141}
selected=gp[-192:];segments=[]
for i,want in enumerate(['control','candidate','candidate','control']*3):
 part=selected[i*16:(i+1)*16];assert len(part)==16 and all(names[r['Op Type']]==want for r in part)
 assert all(int(r['Parallel Engines'])==24 and float(r['Duration (us)'])>0 for r in part)
 segments.append(dict(segment=i,trial=i//4,variant=want,replays=16,median_gp_us=statistics.median(float(r['Duration (us)']) for r in part)))
result=dict(scope='Actual physical GP envelope in final ABBA replay phase; excludes standalone/capture/route mutations',csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),segments=segments,variants={},physical_HBM_measured=False,clock_measured=False,model_gain_measured=False)
chosen={(r['Graph Name'],r['Start time of node']) for r in selected};envelopes=collections.defaultdict(list)
for graph in {r['Graph Name'] for r in selected}:
 gr=sorted([r for r in rows if r['Graph Name']==graph],key=lambda r:float(r['Start time of node']))
 counts=collections.Counter(re.sub(r' iter \d+$','',r['Node Name']) for r in gr);n=len(counts);assert set(counts.values())=={140}
 for i in range(0,len(gr),n):
  chunk=gr[i:i+n];assert set(re.sub(r' iter \d+$','',r['Node Name']) for r in chunk)==set(counts)
  row=next(r for r in chunk if r['Op Type'] in names)
  if (graph,row['Start time of node']) in chosen:
   envelopes[names[row['Op Type']]].append(max(float(r['End time of node']) for r in chunk)-min(float(r['Start time of node']) for r in chunk))
for variant in names.values():
 part=[r for r in selected if names[r['Op Type']]==variant];dur=[float(r['Duration (us)']) for r in part]
 assert len(part)==len(envelopes[variant])==96
 result['variants'][variant]=dict(samples=len(part),median_gp_us=statistics.median(dur),min_gp_us=min(dur),max_gp_us=max(dur),median_core_average_us=statistics.median(float(r['Avg single engine (us)']) for r in part),median_until_msg_write_us=statistics.median(float(r['TPC Duration until MSG_WRITE (us)']) for r in part),median_recipe_envelope_us=statistics.median(envelopes[variant]),all_24_cores=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['variants']))

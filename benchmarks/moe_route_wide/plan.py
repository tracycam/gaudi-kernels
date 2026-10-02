"""CPU-only geometry, input identity, registration and complete ABBA plan."""
import hashlib,json,math
from pathlib import Path
INPUT_SHA='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
VARIANTS=('broadcast','c32','c64','c128')
STATES=('original','route_reverse','input_changed','zero_routes','restored')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_runtime(path):
 path=Path(path);p=json.loads((path/'plan.json').read_text());assert p['format']=='gk-moe-compact8-runtime-v1'
 for name,entry in p['files'].items():assert sha(path/name)==entry['sha256'],name
 # Only the original broadcast databases participate; optional compact/folded
 # libraries remain validated on disk but never registered in this experiment.
 expected=['executor/tpc/libnative_tpc.so','tpc/libbatch_tpc.so','precision-tpc/libprecision_tpc.so']
 assert all(v in p['tpc_libraries']for v in expected)
 return p,expected

def schedule(trials,anchor='broadcast'):
 assert type(trials)is int and 1<=trials<=6
 assert anchor in ('broadcast','c32')
 pairs=('c32','c64','c128') if anchor=='broadcast' else ('c64','c128')
 return[dict(pair=pair,trial=t,arm=i,variant=v)for pair in pairs for t in range(trials)for i,v in enumerate((anchor,pair,pair,anchor))]

def complete(records,trials,anchor='broadcast'):
 want=schedule(trials,anchor);assert len(records)==len(want),'truncated timing matrix'
 for got,w in zip(records,want):
  assert all(got[k]==v for k,v in w.items())and got['output_bits_equal']and got['consumer_bits_equal']and got['stable_logical_handles']
  assert all(math.isfinite(got[k])and got[k]>0 for k in('event_us','wall_us'))and .5<got['event_us']/got['wall_us']<1.2
 return True

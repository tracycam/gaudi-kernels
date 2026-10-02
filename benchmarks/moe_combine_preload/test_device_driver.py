"""CPU-only control flow tests; no Habana imports or devices."""
import ast,types,unittest
from pathlib import Path
from plan import ARMS,STATES,VARIANTS,schedule,complete

class DriverTests(unittest.TestCase):
 def test_abba_all_shapes_states(self):
  rows=schedule((1,2,8),3);self.assertEqual(len(rows),108)
  for n in(1,2,8):
   for state in STATES:
    for trial in range(3):self.assertEqual(tuple(r['variant']for r in rows if(r['rows'],r['state'],r['trial'])==(n,state,trial)),ARMS)
 def valid(self):return[dict(r,event_us=50.,wall_us=55.,output_bits_equal=True,stable_logical_handles=True,output_sha256=['a'*64,'b'*64])for r in schedule((1,2,8),1)]
 def test_complete_keeps_stall(self):
  r=self.valid();r[5].update(event_us=5000.,wall_us=5010.);self.assertTrue(complete(r,(1,2,8),1));self.assertEqual(len(r),36)
 def test_truncated_rejected(self):
  with self.assertRaises(AssertionError):complete(self.valid()[:-1],(1,2,8),1)
 def test_reordered_or_duplicate_rejected(self):
  for mutate in(lambda r:r.__setitem__(0,r[1]),lambda r:r.reverse()):
   r=self.valid();mutate(r)
   with self.assertRaises(AssertionError):complete(r,(1,2,8),1)
 def test_invalid_measurement_or_output_rejected(self):
  for field,value in(('event_us',0),('event_us',float('nan')),('wall_us',float('inf')),('event_us',2),('output_bits_equal',False),('stable_logical_handles',False),('output_sha256',[])):
   r=self.valid();r[0][field]=value
   with self.assertRaises(AssertionError):complete(r,(1,2,8),1)
 def test_shape_domain(self):
  for rows,trials in(((1,1),1),((16,),1),((True,),1),((1,),0),((1,),6)):
   with self.assertRaises(AssertionError):schedule(rows,trials)
 def test_actual_chain_fixed_down_only_changes_combine(self):
  source=Path(__file__).with_name('device_chain.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='chain')
  runs={}
  for arm in VARIANTS:
   log=[]
   class Tensor:
    def __add__(self,v):log.append('add');return self
    def __mul__(self,v):log.append('mul');return self
    def clone(self):log.append('clone');return self
    def to(self,v):log.append('to');return self
    def float(self):log.append('float');return self
   t=Tensor()
   def op(name):
    def call(*args):log.append(name);return t
    return call
   ops=types.SimpleNamespace(gaudi_gp_scale_tail=types.SimpleNamespace(gp=op('gp')),
    unified_batch=types.SimpleNamespace(gate=op('gate')),
    gaudi_down_activation=types.SimpleNamespace(broadcast=op('deployed289')),
    gaudi_combine_preload=types.SimpleNamespace(control=op('control'),unroll8=op('unroll8')))
   def topk(*args,**kw):log.append('topk');return t,t
   torch=types.SimpleNamespace(ops=ops,topk=topk,softmax=lambda *a,**kw:t,int32='i32',float32='f32')
   def combine(p,r,d,h,k):
    self.assertIs(d,t);self.assertEqual((h,k),(6144,8));r.float().clone();log.append('deployed');return t
   ns=dict(torch=torch,owners={1:dict(x=t,logits=t,gain=t)},gp=t,gs=t,dp=t,ds=t,table=t,directions=t,production_combine=combine)
   exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual chain>','exec'),ns);ns['chain'](1,arm)
   self.assertEqual([x for x in log if x in('gp','gate','deployed289',arm)],['gp','gate','deployed289',arm]);self.assertEqual(log.count('clone'),4)
   runs[arm]=['combine'if x==arm else x for x in log]
  self.assertEqual(runs['deployed'],runs['control']);self.assertEqual(runs['deployed'],runs['unroll8'])
 def test_numerics_precede_all_timing(self):
  source=Path(__file__).with_name('device_chain.py').read_text()
  self.assertLess(source.index('numeric_complete=True'),source.index('for arm in timing_plan:'))
  self.assertLess(source.index("torch.save(fixtures[n],dest/'inputs.pt')"),source.index('with torch.hpu.graph(graph'))
  self.assertIn("'zero_routes':dict",source)
  self.assertIn('output_sha256=output_hashes(actual)',source)
  self.assertNotIn('gaudi_down_scale_tail',source)

if __name__=='__main__':unittest.main()

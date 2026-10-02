"""Finite CPU orchestration checks, independent of HPU or fixture availability."""
import ast,unittest
from pathlib import Path
from plan import schedule,complete
class Tests(unittest.TestCase):
 def records(self):return[dict(r,event_us=6000.,wall_us=6050.,output_bits_equal=True,consumer_bits_equal=True,stable_logical_handles=True)for r in schedule(3)]
 def test_abba(self):
  rows=schedule(3);self.assertEqual(len(rows),36)
  for pair in('c32','c64','c128'):
   for trial in range(3):self.assertEqual([r['variant']for r in rows if r['pair']==pair and r['trial']==trial],['broadcast',pair,pair,'broadcast'])
 def test_all_stalls_retained(self):
  r=self.records();r[0].update(event_us=60000.,wall_us=60500.);self.assertTrue(complete(r,3))
 def test_direct_c32_pairing(self):
  rows=schedule(6,'c32');self.assertEqual(len(rows),48)
  for pair in ('c64','c128'):
   for trial in range(6):self.assertEqual([r['variant']for r in rows if r['pair']==pair and r['trial']==trial],['c32',pair,pair,'c32'])
  records=[dict(r,event_us=5000.,wall_us=5050.,output_bits_equal=True,consumer_bits_equal=True,stable_logical_handles=True)for r in rows]
  self.assertTrue(complete(records,6,'c32'))
  with self.assertRaises(AssertionError):complete(records,6)
 def test_bad_or_incomplete_rejected(self):
  cases=[self.records()[:-1],self.records()[::-1]]
  for key,value in(('event_us',float('nan')),('event_us',0.),('output_bits_equal',False),('consumer_bits_equal',False),('stable_logical_handles',False)):
   r=self.records();r[0][key]=value;cases.append(r)
  for r in cases:
   with self.assertRaises(AssertionError):complete(r,3)
 def test_no_acceptance_and_original_persisted(self):
  s=Path(__file__).with_name('device_chain.py').read_text();ast.parse(s)
  self.assertIn('candidate_accepted=False',s);self.assertNotIn('candidate_accepted=True',s)
  self.assertLess(s.index("torch.save(fixtures,out/'inputs.pt')"),s.index('with torch.hpu.graph'))
  self.assertLess(s.index("result['numeric_complete']=True"),s.index('for arm in arms:'))
  self.assertIn('CROSS_C_FP32_AUDIT_REQUIRED',s)
  self.assertIn("mode='broadcast'",s)
  self.assertNotIn("mode='compact'",s)
if __name__=='__main__':unittest.main()

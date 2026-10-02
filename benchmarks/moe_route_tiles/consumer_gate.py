#!/usr/bin/env python3
"""Colored primitive callbacks test dataflow only, not MXFP4 or SiLU accuracy."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from planner import Capacity,assign,fixture
from functional_reference import Interface,execute
from cpu_gate import ordered_combine

def f32(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
def bf16(values):
 b=values.astype(np.float32).view(np.uint32);return ((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)

def gp(a,experts):
 p=np.repeat(f32(a)[...,:1],8,axis=-1)+np.arange(8,dtype=np.float32)[None,None,:]
 p+=np.maximum(experts,0).astype(np.float32)[:,None,None]
 p[experts<0]=np.nan
 return p.astype(np.float32)

def fake_gate(p,valid):
 # An opaque BF16 boundary stand-in. Deliberately does not pretend to be SiLU.
 y=bf16(p[...,:4])
 for local,n in enumerate(valid):y[local,int(n):]=0
 return y

def down(g,experts,n0,width):
 p=(f32(g)[...,:1]+np.arange(n0,n0+width,dtype=np.float32)[None,None,:]/16
    +np.maximum(experts,0).astype(np.float32)[:,None,None]/4).astype(np.float32)
 p[experts<0]=np.nan
 return p

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
 interface=Interface(input_width=8,gate_width=4,output_width=16,n_tile=8)
 bad_interfaces=[replace(interface,**{name:value}) for name in Interface.__dataclass_fields__ for value in (0,-1,1.5,True)]
 bad_interfaces += [replace(interface,gp_batch=32),replace(interface,down_batch=3),replace(interface,output_width=17)]
 for bad in bad_interfaces:
  try:bad.check()
  except ValueError:pass
  else:raise AssertionError('invalid interface accepted')
 interface.check();Interface().check()
 for t in (2,32,128,512,513):
  for kind in ('uniform','hot','skew'):
   ids=fixture(t,8,384,kind);plan=assign(ids,Capacity(t,8,384,16));x=bf16((np.arange(t*8).reshape(t,8)%31-15).astype(np.float32)/4)
   routing=((np.arange(t*8).reshape(t,8)%7-3)/8).astype(np.float32)
   y,stats=execute(x,ids,routing,plan,gp=gp,literal_gate=fake_gate,down=down,combine=ordered_combine,interface=interface)
   # Independent per-route callback sequence, no tile metadata or inverse.
   expert=np.array(ids,np.int32).reshape(-1);a_route=np.repeat(x,8,axis=0)[:,None,:]
   p_route=gp(a_route,expert);g_route=fake_gate(p_route,np.ones(t*8,np.int32))
   want=ordered_combine(down(g_route,expert,0,16).reshape(t,8,16),routing)
   assert np.array_equal(y.view(np.uint32),want.view(np.uint32))
   active=sum(n>0 for n in plan.tile_valid_rows)
   assert stats['gp_active_weight_tiles']==active
   assert stats['down_active_weight_tile_width']==active*interface.output_width
   assert stats['max_down_tile_payload_bytes']==plan.capacity.max_tiles*plan.capacity.rows*interface.n_tile*4
   assert stats['gate_concats']<stats['down_fragments'] # hoisted across both N tiles
   records.append(dict(tokens=t,distribution=kind,all_bits_equal=True,checked_fp32_words=y.size,**stats))
 # Explicit FP32 slot-order witness across a nonintegral T513 final tile.
 t=513;ids=fixture(t,8,384,'hot');plan=assign(ids,Capacity(t,8,384,16))
 x=np.zeros((t,8),np.uint16);routing=np.ones((t,8),np.float32)
 def cancellation_down(g,experts,n0,width):
  p=np.zeros((len(experts),g.shape[1],width),np.float32)
  if n0==0:
   for index,expert in enumerate(experts):p[index,:,0]={383:2**24,382:1,381:-2**24,380:1}.get(int(expert),0)
  return p
 y,_=execute(x,ids,routing,plan,gp=gp,literal_gate=fake_gate,down=cancellation_down,combine=ordered_combine,interface=interface)
 wrong,_=execute(x,ids,routing,plan,gp=gp,literal_gate=fake_gate,down=cancellation_down,
   combine=lambda p,w:ordered_combine(p[:,::-1],w[:,::-1]),interface=interface)
 assert np.all(y[:,0]==1) and np.all(wrong[:,0]==2)
 precision_witness={'tokens':513,'last_expert_tile_valid_rows':1,'actual_active_tiles':264,
   'ordered_slot_terms':[2**24,1,-2**24,1,0,0,0,0],'expected_fp32':1,'reversed_order_mutant':2,
   'mutant_rejected':True,'checked_columns':513,'scope':'FMA slot order only; legal internal MME tree differences remain allowed'}
 report={'status':'CPU_FUNCTIONAL_DATAFLOW_PASS','device_implemented':False,'precision_qualified':False,
         'slot_precision_witness':precision_witness,'interface_guard_cases':len(bad_interfaces),
         'scope':'colored callback dataflow, original BF16 payload and slot-order addressing; no MXFP4 MAC or literal SiLU numeric claim','records':records}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cases':len(records),'words':sum(r['checked_fp32_words'] for r in records),'status':report['status']}))
if __name__=='__main__':main()

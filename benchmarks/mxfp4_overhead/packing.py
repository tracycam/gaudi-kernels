"""Roundtrip checks for the reusable versioned load-time packing API."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"python"))
from gaudi_kernels.mxfp4_prepared_gemv import PreparedN512, prepare, pack, unpack
if __name__=='__main__':
 rng=np.random.default_rng(2709)
 for n,k in [(1,1),(513,257),(512,6144),(6144,256)]:
  r=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8);s=rng.integers(2,253,(n,(k+31)//32),dtype=np.uint8)
  if k%2:r[:,-1]&=15
  w,t=pack(r,s,k);rr,ss=unpack(w,t,n,k);assert np.array_equal(r,rr) and np.array_equal(s,ss)
  print('PASS',n,k,'original_bytes',r.nbytes+s.nbytes,'prepared_bytes',w.nbytes+t.nbytes)

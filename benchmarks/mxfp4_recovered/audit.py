"""Offline proof checks for recovered-run scheduling and conservative bias guard."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
checked=0
for exponent in range(255):
 for mantissa in [0,1,2,3,0x3fffff,0x400000,0x7fffff]:
  m=mantissa|(0x800000 if exponent else 0);e=exponent or 1
  if not m:continue
  exact_lo=e-150+(m&-m).bit_length()-1;exact_hi=e-150+m.bit_length()-1
  assert e-150<=exact_lo<=exact_hi<=e-127;checked+=1
ownership=[]
for N in [512,6144,32768]:
 seen=set();bytes_seen=set()
 for block in range(N//256):
  for pair in range(block*128,(block+1)*128):
   n0=(pair//256)*512+(pair%256//128)*256+pair%128
   addresses=[]
   for n in (n0,n0+128):
    assert block*256<=n<(block+1)*256 and n not in seen;seen.add(n)
    native=(n%512//128)*128+2*(n%64)+(n%128)//64
    addresses.append(((n//512)*256+(native//256)*128+native%128,4*((native%256)//128)))
   assert addresses[0][0]==addresses[1][0] and addresses[0][1]!=addresses[1][1]
   assert addresses[0][0] not in bytes_seen;bytes_seen.add(addresses[0][0])
 assert len(seen)==N and len(bytes_seen)==N//2
 ownership.append({'N':N,'output_owners':len(seen),'unique_packed_bytes_per_K':len(bytes_seen)})
text=(a.build_root/'v1/exact/mx_dispatch_history.text').read_bytes()
assert text==(a.build_root/'v2/exact/mx_dispatch_history.text').read_bytes()
result={'all_pass':True,'bias_bounds_checked':checked,'bound':'nonzero bias effective_exponent-150 <= exact least bit; exact highest bit <= effective_exponent-127','finish_ownership':ownership,'conditional_history_text_identical_v1_v2':True,'conditional_history_text_sha256':hashlib.sha256(text).hexdigest(),'physical_hbm_counter_claim':False}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

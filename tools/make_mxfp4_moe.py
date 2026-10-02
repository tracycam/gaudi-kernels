"""Wrap the proven K32 scheduled body; keep ordinary FP32 arithmetic order."""
from pathlib import Path
NOP='nop; nop; nop; nop\n'
def make(source,destination,fused,narrow=False):
 text=Path(source).read_text();head=text.split('loop S3, S4, 1, <, .LBB0_7')[0]
 # Preserve scalar parameters before the historical descriptor prologue.
 head=head.replace('main:\n','main:\nnop; mov.i32 S30, S0; nop; nop\nnop; mov.i32 S31, S1; nop; nop\nnop; mov.i32 S23, S2; nop; nop\nnop; mov.i32 S22, S3; nop; nop\n')
 head+='ld_l mmio S29, 0x150; nop; nop; nop\n'+NOP*8
 out=[head,'loop S3, S4, 1, <, .task_end\n',NOP*2]
 def ins(s,delay=6):out.append('nop; '+s+'; nop; nop\n'+NOP*delay)
 def raw(s,delay=6):out.append(s+'\n'+NOP*delay)
 def divide(value,divisor,reciprocal,quotient,remainder):
  ins(f'mul.u32 upper32 {quotient}, {value}, {reciprocal}')
  ins(f'mul.i32 {remainder}, {quotient}, {divisor}')
  ins(f'cmp_grt.u32 SP2, {remainder}, {value}')
  ins(f'sub.i32 {quotient}, {quotient}, 0x1, SP2')
  ins(f'cmp_eq.i32 SP2, {divisor}, 0x1')
  ins(f'mov.i32 {quotient}, {value}, SP2')
  ins(f'mul.i32 {remainder}, {quotient}, {divisor}')
  ins(f'sub.i32 {remainder}, {value}, {remainder}')
 divide('S32','S30','S31','S24','S25') # quotient token(fused) or route(direct), remainder Nblock
 if fused:
  if narrow:
   for v in range(4,8):raw(f'mov.f32 V{v}, 0x0; nop; nop; nop',0)
  else:
   raw('mov.f32 V16, 0x0; nop; nop; nop')
   for v in range(8):raw(f'nop; nop; nop; st_l_v {v*256}, V16',0)
  out.append(NOP*6+'loop 0, S29, 1, <, .routes_end\n'+NOP*2)
  ins('mul.i32 S21, S24, S29');ins('add.i32 S21, S21, S33')
  slot,token,route='S33','S24','S21'
 else:
  divide('S24','S29','S22','S21','S20');slot,token,route='S20','S21','S24'
 raw('set_indx I0, b11111, 0x0; nop; nop; nop')
 ins(f'set_indx I0, b00001, {slot}');ins(f'set_indx I0, b00010, {token}')
 raw('nop; nop; nop; gen_addr AD0, 0x4, I0')
 raw('ld_g S10, AD0; nop; nop; nop',10)
 ins('cmp_geq.i32 SP1, S10, 0x0');ins('cmp_less.i32 SP2, S10, S23')
 ins('and.b SP1, SP1, SP2')
 if fused:
  raw('nop; nop; nop; gen_addr AD0, 0x5, I0')
  raw('ld_g S26, AD0; nop; nop; nop',10)
  ins('mov.f32 S26, 0x0, !SP1')
 ins('mul.i32 S10, S10, S30');ins('add.i32 S10, S10, S25')
 if narrow:
  ins('and.i32 S28, S10, 0x1');ins('shr.u32 S10, S10, 0x1')
  ins('mul.i32 S1, S10, S0');ins('shl.i32 S1, S1, 0x1');ins('add.i32 S1, S1, S28')
  ins('mul.i32 S3, S10, S2');ins('shl.i32 S3, S3, 0x1');ins('add.i32 S3, S3, S28')
 else:ins('mul.i32 S1, S10, S0');ins('mul.i32 S3, S10, S2')
 ins('mov.i32 S4, S1')
 for coord in ['I5','I6','I7']:raw(f'set_indx {coord}, b11111, 0x0; nop; nop; nop')
 ins(f'set_indx I6, b00010, {route}')
 for v in range(4 if narrow else 8):raw(f'mov.f32 V{v}, 0x0; nop; nop; nop',0)
 out.append(NOP*6+'loop 0, S2, 1, <, .groups_end, SP1\n'+NOP*2)
 body=text.split('.LBB0_3:\n')[1].split('.LBB0_6:')[0]
 if fused:body=body.replace('S33','S34')
 if narrow:
  import re
  lines=[]
  for line in body.splitlines():
   slots=[x.strip() for x in line.split(';')]
   for i,op in enumerate(slots):
    if (re.search(r'\b(?:D8|D10|V8|V9|V10|V11|V36|V37)\b',op)
      or re.search(r'lookup_2c.*\b(?:D20|D26|D32)\b',op)
      or re.search(r'mac.f32 V[4567],',op)
      or 'convert.bf16 all_lanes target_type=fp32 rhne D32' in op
      or 'convert.bf16 all_lanes target_type=fp32 rhne D34' in op):slots[i]='nop'
    slots[i]=slots[i].replace('add.i32 b00010 I5, 0x1, I5','add.i32 b00010 I5, 0x2, I5').replace('add.i32 S4, S4, 0x20','add.i32 S4, S4, 0x40')
    if 'set_indx I7, b00010, S34' in slots[i]:slots[i]=slots[i].replace('S34','S6')
   lines.append('; '.join(slots))
  body='nop; shl.i32 S6, S34, 0x1; nop; nop\n'+NOP*6+'\n'.join(lines)+'\n'
 out.append(body+'.groups_end:\n'+NOP*8)
 if fused:
  if narrow:
   for v in range(4):raw(f'nop; nop; mac.f32 V{v+4}, V{v}, S26; nop',1)
  else:
   for v in range(8):
    raw(f'ld_l_v V16, {v*256}; nop; nop; nop')
    raw(f'nop; nop; mac.f32 V16, V{v}, S26; nop')
    raw(f'nop; nop; nop; st_l_v {v*256}, V16',0)
  out.append('.routes_end:\n'+NOP*8)
  row='S24';regs=[4,5,6,7] if narrow else [0,1,2,3,6,7,4,5]
 else:row='S24';regs=[4,5,6,7] if narrow else [0,1,2,3,6,7,4,5]
 raw('set_indx I4, b11111, 0x0; nop; nop; nop')
 ins('shl.i32 S4, S25, '+('0x8' if narrow else '0x9'));ins('set_indx I4, b00001, S4');ins(f'set_indx I4, b00010, {row}')
 for v in regs:
  if fused and not narrow:raw(f'ld_l_v V16, {v*256}; nop; nop; nop')
  out.append(f'nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x6, I4, V{16 if fused and not narrow else v}\n')
 out.append('.task_end:\n'+NOP*8+'nop; halt; halt; nop\n'+NOP*8)
 Path(destination).write_text(''.join(out))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--fused',action='store_true');a=p.parse_args();make(a.source,a.destination,a.fused)

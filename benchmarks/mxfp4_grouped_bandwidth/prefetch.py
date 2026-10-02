"""Pinned production GP schedule, with guarded next-K32 prefetch.

The control adapts activation addressing from token to grouped expert slot.
The candidate bootstraps two weight-index vectors and activation once per
task, then prefetches the next group in dead registers in the scale tail.
Last-group tensor prefetch is predicated OFF: no extra original weight read.
"""
import hashlib,json
from pathlib import Path


def transform(source,prefetch,safe=False):
    old='set_indx I6, b00010, S13'
    assert source.count(old)==1
    source=source.replace(old,'set_indx I6, b00010, S11')
    if not prefetch:return source,{'grouped_activation_coordinate':True,'prefetch':False}
    prefix,body=source.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
    old=[[x.strip()for x in line.split(';')]for line in body.splitlines()]
    assert len(old)==247 and all(len(row)==4 for row in old)
    assert old[18][0]=='ld_tnsr UNPCK_8_TO_16 unpack V38, 0x1, I7'
    assert old[34][0]=='ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5'
    nop=['nop']*4
    rows=[nop.copy()for _ in range(9)]+[row.copy()for row in old[34:]]
    def put(i,slot,op):
        assert rows[i][slot]=='nop',(i,slot,rows[i][slot]);rows[i][slot]=op
    put(0,0,'mov.f32 V14, 0x0');put(0,2,'mov.f32 V15, 0x0')
    put(0,1,'cmp_less.i32 SP3, S33, 0x3f')
    put(1,0,'ld_tnsr UNPCK_8_TO_16 unpack V38, 0x1, I7');put(1,2,'mov.f32 V12, 0x0')
    put(2,0,'mov.f32 V10, 0x0');put(2,2,'shuffle.u8 V36, V39, V30')
    put(3,0,'ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5');put(3,1,'add.i32 b00010 I5, 0x1, I5');put(3,2,'mov.f32 V11, 0x0')
    put(4,0,'mov.f32 V8, 0x0');put(4,2,'mov.f32 V9, 0x0')
    put(5,0,'lookup_2c BV16 lut_ptr D18, V16, S27');put(5,2,'mov.f32 V13, 0x0')
    # LOAD-slot immediate between the two LUT issues corrupts activation k2.
    # Place the same constant in the vacant VPU slot; retain its issue packet.
    put(6,2,'mov.i32 V34, 0x85848584')
    put(7,0,'lookup_2c BV16 lut_ptr D20, V17, S27')
    put(8,2,'shuffle.u8 V37, V39, V31')
    # These registers are dead after the final group MACs. Only the next
    # group's first two weight vectors are read, and only when that group exists.
    put(194,1,'add.i32 b00001 I6, 0x20, I6')
    put(201,0,'ld_tnsr V39, 0x2, I6, SP3')
    put(204,0,'ld_tnsr UNPCK_8_TO_16 unpack V24, 0x0, I5, SP3');put(204,1,'add.i32 b00010 I5, 0x1, I5')
    put(205,0,'ld_tnsr UNPCK_8_TO_16 unpack V26, 0x0, I5, SP3');put(205,1,'add.i32 b00010 I5, 0x1, I5')
    put(207,2,'mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff')
    put(210,1,'set_indx I7, b00001, 0x0')
    put(211,1,'add.i32 b00010 I7, 0x1, I7')
    put(211,2,'convert.u8 all_lanes target_type=uint16 rhne D16, V24')
    put(212,2,'convert.u8 all_lanes target_type=uint16 rhne D22, V26')
    put(218,0,'mov.i32 V30, 0x81808180');put(219,0,'mov.i32 V31, 0x83828382')
    if safe:
        # Isolate tensor load/convert readiness: retain the same registers and
        # arithmetic, space the two loads and postpone both conversions.
        rows[205][0]=rows[205][1]='nop'
        put(208,0,'ld_tnsr UNPCK_8_TO_16 unpack V26, 0x0, I5, SP3');put(208,1,'add.i32 b00010 I5, 0x1, I5')
        rows[211][2]=rows[212][2]='nop'
        rows += [['nop','nop','convert.u8 all_lanes target_type=uint16 rhne D16, V24','nop'],['nop','nop','convert.u8 all_lanes target_type=uint16 rhne D22, V26','nop']]
    # Bootstrap the same incoming registers before the hardware K32 loop.
    bootstrap=[nop.copy()for _ in range(36)]
    def boot(i,slot,op):assert bootstrap[i][slot]=='nop';bootstrap[i][slot]=op
    boot(0,1,'set_indx I5, b00001, 0x0');boot(0,2,'shl.i32 S7, S15, 0x5')
    # Scalar shift belongs to SPU, not VPU.
    bootstrap[0][2]='nop';boot(1,1,'shl.i32 S7, S15, 0x5')
    boot(2,1,'set_indx I5, b00010, S1')
    boot(3,1,'set_indx I7, b00001, 0x0');boot(4,1,'set_indx I7, b00010, S3')
    boot(6,1,'set_indx I6, b00001, S7')
    boot(10,0,'ld_tnsr UNPCK_8_TO_16 unpack V24, 0x0, I5');boot(10,1,'add.i32 b00010 I5, 0x1, I5')
    boot(11,0,'ld_tnsr UNPCK_8_TO_16 unpack V26, 0x0, I5');boot(11,1,'add.i32 b00010 I5, 0x1, I5')
    boot(14,0,'ld_tnsr V39, 0x2, I6')
    boot(18,2,'convert.u8 all_lanes target_type=uint16 rhne D16, V24')
    boot(19,2,'convert.u8 all_lanes target_type=uint16 rhne D22, V26')
    boot(22,2,'mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff')
    boot(24,0,'mov.i32 V30, 0x81808180');boot(25,0,'mov.i32 V31, 0x83828382')
    if safe:
        bootstrap[11][0]=bootstrap[11][1]=bootstrap[14][0]='nop'
        boot(14,0,'ld_tnsr UNPCK_8_TO_16 unpack V26, 0x0, I5');boot(14,1,'add.i32 b00010 I5, 0x1, I5')
        boot(18,0,'ld_tnsr V39, 0x2, I6')
        bootstrap[18][2]=bootstrap[19][2]=bootstrap[22][2]='nop'
        boot(24,2,'convert.u8 all_lanes target_type=uint16 rhne D16, V24');boot(25,2,'convert.u8 all_lanes target_type=uint16 rhne D22, V26')
        boot(26,2,'mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff')
    loop='loop 0, S2, 1, <, .LBB0_6, SP1\n';assert prefix.count(loop)==1
    prefix=prefix.replace(loop,'\n'.join('; '.join(r)for r in bootstrap)+'\n'+loop)
    old_macs=[r[2]for r in old if r[2].startswith('mac.')]
    assert old_macs==[r[2]for r in rows if r[2].startswith('mac.')]
    memory=[(i,r[0].split()[0])for i,r in enumerate(rows)if r[0].startswith(('ld_tnsr','lookup_2c'))]
    gaps=[(i,j)for (i,a),(j,b)in zip(memory,memory[1:])if 'lookup_2c'in(a,b)]
    assert all(j-i>=2 for i,j in gaps),gaps
    return prefix+'.LBB0_3:\n'+'\n'.join('; '.join(r)for r in rows)+'\n.LBB0_6:'+suffix,dict(
        prefetch=True,safe=safe,mask_k2_uses_vpu_slot=True,old_packets=247,new_packets=len(rows),bootstrap_packets=len(bootstrap),
        original_MAC_sequence_unchanged=True,weight_prefetch_last_group_predicated_off=True,
        lookup_neighbor_gaps=gaps,device_verified=False)

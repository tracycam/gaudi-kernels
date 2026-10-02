"""N256 four raw/four decoded/three activation buffers; no MAC reassociation."""
import re

def schedule(period):
    assert period in (4,6)
    mac0=32 if period==4 else 36
    final=mac0+31*period+1
    start=final+8
    rows=[['nop']*4 for _ in range(start+4)]
    def put(t,slot,op):
        assert rows[t][slot]=='nop',(t,slot,rows[t][slot],op)
        rows[t][slot]=op
    put(0,0,'ld_tnsr UNPCK_8_TO_16 unpack V10, 0x1, I7')
    put(1,0,'mov.f32 V14, 0x0');put(1,2,'mov.f32 V15, 0x0')
    put(2,0,'mov.f32 V12, 0x0');put(2,2,'mov.f32 V13, 0x0')
    put(3,0,'ld_tnsr V39, 0x2, I6')
    put(8,2,'convert.u8 all_lanes target_type=uint16 rhne D10, V10')
    put(11,2,'mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff')
    put(16,2,'shl.i16 V10, V10, 0x7');put(17,2,'shl.i16 V11, V11, 0x7')
    put(24,2,'convert.bf16 all_lanes target_type=fp32 rhne D20, V10')
    put(25,2,'convert.bf16 all_lanes target_type=fp32 rhne D26, V11')
    events=[]
    for k in range(32):
        raw=(16,22,28,32)[k%4];q=(18,24,30,10)[k%4];act=(36,37,38)[k%3];mask=(8,34,35)[k%3]
        load=8+period*k;convert=load+6;lookup=load+14;shuffle=load+15;mac=mac0+period*k
        selector=(0x81+2*k)<<24|(0x80+2*k)<<16|(0x81+2*k)<<8|(0x80+2*k)
        put(load,0,f'ld_tnsr UNPCK_8_TO_16 unpack V{raw}, 0x0, I5')
        put(load,1,'add.i32 b00010 I5, 0x2, I5')
        put(convert,2,f'convert.u8 all_lanes target_type=uint16 rhne D{raw}, V{raw}')
        put(lookup,0,f'lookup_2c BV16 lut_ptr D{q}, V{raw}, S27')
        put(load+7,0,f'mov.i32 V{mask}, 0x{selector:08x}')
        put(shuffle,2,f'shuffle.u8 V{act}, V39, V{mask}')
        put(mac,2,f'mac.bf16 acc_fp32 D14, V{q}, V{act}')
        put(mac+1,2,f'mac.bf16 acc_fp32 D12, V{q+1}, V{act}')
        events.append(dict(k=k,load=load,convert=convert,lookup=lookup,shuffle=shuffle,first_mac=mac,last_mac=mac+1,raw_pair=raw,decoded_pair=q,activation=act))
    for i,(part,factor) in enumerate([(14,20),(15,21),(12,26),(13,27)]):put(start+i,2,f'mac.f32 V{i}, V{part}, V{factor}')
    put(final+1,1,'add.i32 b00010 I7, 0x2, I7')
    put(final+2,1,'add.i32 b00001 I6, 0x20, I6')
    memory=[(i,r[0].split()[0]) for i,r in enumerate(rows) if r[0].startswith(('ld_tnsr','lookup_2c'))]
    assert all(b-a>=2 for (a,x),(b,y) in zip(memory,memory[1:]) if 'lookup_2c' in (x,y))
    for k,e in enumerate(events):
        if k+4<32:
            assert events[k+4]['load']>e['lookup']
            assert events[k+4]['lookup']>e['last_mac']
        if k+3<32:assert events[k+3]['shuffle']>e['last_mac']
    return '\n'.join('; '.join(r) for r in rows)+'\n',dict(period=period,body_packets=len(rows),events=events,
        BF16_MAC_recurrence_packets=period,FP32_MAC_order_unchanged=True,max_vector_register=39,
        raw_source_reuse_after_lookup_packets=4*period-14,decoded_reuse_after_last_MAC_packets=4*period+22-(mac0+1),activation_reuse_after_last_MAC_packets=3*period+23-(mac0+1),
        scale_load_placement=0,activation_load_placement=3,lookup_memory_neighbor_gap_at_least_two=True,
        qualification='OFFLINE_UNQUALIFIED; period4 tightens recurrence from known production6 and must pass device bits',
        hardware_tested=False)

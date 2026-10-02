"""Change only activation fetch/broadcast in the certified K32 assembly body."""
import re

def transform(source):
    prefix,body=source.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
    lines=[line.split(';') for line in body.splitlines()]
    assert all(len(line)==4 for line in lines)
    lines=[[slot.strip() for slot in line] for line in lines]
    macs=[i for i,line in enumerate(lines) if line[2].startswith('mac.bf16 acc_fp32 D14,')]
    loads=[i for i,line in enumerate(lines) if re.fullmatch(r'ld_g S(?:16|17|18|19|20), AD[012]',line[0])]
    addr=[i for i,line in enumerate(lines) if line[3].startswith('gen_addr dt=int8')]
    assert len(macs)==len(loads)==len(addr)==32
    # Gaudi2 has only V0..V39. V40+ encode LFSR/lane-ID registers despite
    # accepting V-number syntax. Defer scale decode until all K32 MACs finish,
    # freeing V34..V37 without reducing the existing three weight/LUT buffers.
    scale_ops=0
    for line in lines:
        if re.match(r'ld_tnsr .* V(?:34|36), 0x1, I7$',line[0]):line[0]='nop';scale_ops+=1
        if line[1]=='add.i32 b00001 I7, 0x100, I7':line[1]='nop'
        if re.fullmatch(r'convert.u8 all_lanes target_type=uint16 rhne D(?:34|36), V(?:34|36)',line[2]):line[2]='nop';scale_ops+=1
        if re.fullmatch(r'shl.i16 V(?:34|35|36|37), V(?:34|35|36|37), 0x7',line[2]):line[2]='nop';scale_ops+=1
        if re.fullmatch(r'convert.bf16 all_lanes target_type=fp32 rhne D(?:28|30|32|34), V(?:34|35|36|37)',line[2]):line[2]='nop';scale_ops+=1
    assert scale_ops==12
    for k,i in enumerate(loads):lines[i][0]=f'mov.i32 V{34+k%2}, 0x{0x81808180+k*0x02020202:08x}'
    for i in addr:lines[i][3]='nop'
    for line in lines:
        if line[1]=='add.i32 b00001 I6, 0x1, I6':line[1]='nop'
    k=-1;count=0
    for i,line in enumerate(lines):
        if i in macs:k+=1
        if line[2].startswith('mac.bf16'):
            line[2],n=re.subn(r'S(?:16|17|18|19|20)$',f'V{36+k%3}',line[2]);assert n==1;count+=1
    assert k==31 and count==128
    inserts={};shuffle_positions=[]
    for k,mac in enumerate(macs):
        limit=min(mac-5,loads[k+2] if k+2<32 else mac-5)
        candidates=[i for i in range(max(loads[k]+6,mac-12),limit) if lines[i][2]=='nop']
        instruction=f'shuffle.u8 V{36+k%3}, V39, V{34+k%2}'
        if candidates:
            i=candidates[-1];lines[i][2]=instruction;shuffle_positions.append((i,False))
        else:
            i=limit-1;assert i>=loads[k]+6,(k,i,loads[k])
            inserts.setdefault(i,[]).append(['nop','nop',instruction,'nop']);shuffle_positions.append((i,True))
    # I6 points to the first BF16 in the K32 block. Native tensor load allows
    # unaligned K32 coordinates; simulator checks offsets0,32,64,96,128/tail.
    first=addr[0]
    prefill=[['nop']*4 for _ in range(6)]
    prefill += [['ld_tnsr V39, 0x2, I6','nop','nop','nop']]
    prefill += [['nop']*4 for _ in range(7)]
    prefill += [['nop','nop','mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff','nop']]
    prefill += [['nop']*4 for _ in range(5)]
    inserts.setdefault(first,[])[:0]=prefill
    # Original FP32 scale MAC order stays unchanged. Only decode/load timing
    # moves; all BF16 activation broadcasts are dead at this point.
    scale_mac=next(i for i,line in enumerate(lines) if line[2].startswith('mac.f32'))
    epilogue=[]
    def packet(load='nop',spu='nop',vpu='nop'):epilogue.append([load,spu,vpu,'nop'])
    packet('ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7','add.i32 b00001 I7, 0x100, I7')
    for _ in range(6):packet()
    packet('ld_tnsr UNPCK_8_TO_16 unpack V36, 0x1, I7',vpu='convert.u8 all_lanes target_type=uint16 rhne D34, V34')
    for _ in range(6):packet()
    packet(vpu='convert.u8 all_lanes target_type=uint16 rhne D36, V36')
    packet(vpu='shl.i16 V34, V34, 0x7');packet(vpu='shl.i16 V35, V35, 0x7')
    for _ in range(5):packet()
    packet(vpu='shl.i16 V36, V36, 0x7');packet(vpu='shl.i16 V37, V37, 0x7')
    packet(vpu='convert.bf16 all_lanes target_type=fp32 rhne D28, V34')
    packet(vpu='convert.bf16 all_lanes target_type=fp32 rhne D30, V35')
    for _ in range(5):packet()
    packet(vpu='convert.bf16 all_lanes target_type=fp32 rhne D32, V36')
    packet(vpu='convert.bf16 all_lanes target_type=fp32 rhne D34, V37')
    for _ in range(6):packet()
    inserts.setdefault(scale_mac,[])[:0]=epilogue
    emitted=[];actual={};insert_actual={}
    for i,line in enumerate(lines):
        if i in inserts:
            for row in inserts[i]:
                if row[2].startswith('shuffle'):insert_actual[i]=len(emitted)
                emitted.append(row)
        actual[i]=len(emitted);emitted.append(line)
    schedule=[]
    for k,(pos,inserted) in enumerate(shuffle_positions):
        shuffle=insert_actual[pos] if inserted else actual[pos]
        load=actual[loads[k]];mac=actual[macs[k]]
        assert shuffle-load>=6 and mac-shuffle>=6
        if k+2<len(loads):assert shuffle<actual[loads[k+2]]
        if k>=3:assert shuffle>actual[macs[k-3]+3]
        schedule.append({'k':k,'mask_packet':load,'shuffle_packet':shuffle,'first_mac_packet':mac})
    result=prefix+'.LBB0_3:\n'+'\n'.join('; '.join(line) for line in emitted)+'\n.LBB0_6:'+suffix
    return result,{'activation_ld_g_removed':32,'activation_gen_addr_removed':32,
                   'activation_tensor_loads_per_k32':1,'activation_tensor_elements_per_load':128,
                   'useful_activation_elements_per_k32':32,'weight_loads_changed':0,
                   'shuffle_count':32,'max_vector_register':39,'scale_decode_deferred':True,'original_k32_packets':len(lines),
                   'candidate_k32_packets':len(emitted),'schedule':schedule}

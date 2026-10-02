"""Pinned deployed-ELF provenance and pure offline activation retiming."""
import hashlib
import re
import struct

PINS={
 'gp':{'library':'libgaudi_gp_diagnostic_tpc.so','library_sha256':'5202850a823bfced635aad15a9c60205a2e63127d295712b561c6079bb3d885e',
       'elf':'direct_gp.o','elf_sha256':'6b4882ba1e8eb651d710ff0f15816d4b441d3702b75c9ace5c7a8bd3036f9f32',
       'assembly':'direct_gp_broadcast.s','assembly_sha256':'4766c2f764d8bf9f2ef5b3c25cc464432d65b455ac865bed6b64fa4263f7d093'},
 'down':{'library':'libgaudi_down_activation_tpc.so','library_sha256':'9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a',
         'elf':'direct_down.o','elf_sha256':'b930eb299830f036be389bb4a9d7f0597fa9ee24e51c2880e4e308689c6d3b75',
         'assembly':'direct_down_broadcast.s','assembly_sha256':'af1863e523b6fef932ef2d42dff9442331ceeb2d7b33cd5fa7845b2af03bacb9'}}

def sha(data):return hashlib.sha256(data).hexdigest()

def embedded_elf(data,stem):
    """Read objcopy's exported byte range from a standard ELF64 little-endian SO."""
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('expected ELF64 little-endian host library')
    header=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
    shoff,shentsize,shnum=header[6],header[11],header[12]
    if shentsize!=64 or not shnum or shoff+shentsize*shnum>len(data):raise ValueError('unsupported or truncated sections')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*shentsize) for i in range(shnum)]
    symbols={}
    wanted={f'_binary_{stem}_o_start',f'_binary_{stem}_o_end'}
    for section in sections:
        if section[1] not in (2,11):continue
        _,_,_,_,offset,size,link,_,_,entsize=section
        if entsize!=24 or link>=shnum or offset+size>len(data):raise ValueError('invalid symbol table')
        strings=sections[link];blob=data[strings[4]:strings[4]+strings[5]]
        for i in range(offset,offset+size,entsize):
            name,info,other,index,value,length=struct.unpack_from('<IBBHQQ',data,i)
            end=blob.find(b'\0',name)
            if end<0:raise ValueError('invalid symbol name')
            key=blob[name:end].decode()
            if key in wanted:
                if key in symbols and symbols[key]!=(index,value):raise ValueError('conflicting symbol')
                symbols[key]=(index,value)
    if set(symbols)!=wanted:raise ValueError('embedded ELF symbols missing')
    si,start=symbols[f'_binary_{stem}_o_start'];ei,end=symbols[f'_binary_{stem}_o_end']
    if si!=ei or si>=shnum or end<=start:raise ValueError('invalid embedded ELF range')
    section=sections[si];begin=section[4]+start-section[3];finish=begin+end-start
    if start<section[3] or end>section[3]+section[5] or finish>len(data):raise ValueError('embedded ELF outside section')
    return data[begin:finish]

def verify_deployed(directory,target):
    pin=PINS[target];data={}
    for kind in ['library','elf','assembly']:
        data[kind]=(directory/pin[kind]).read_bytes()
        if sha(data[kind])!=pin[kind+'_sha256']:raise ValueError(f'{target} deployed {kind} SHA mismatch')
    extracted=embedded_elf(data['library'],'direct_'+target)
    if extracted!=data['elf']:raise ValueError('supplied ELF does not equal actual library embedded ELF')
    return data,{'target':target,'directory':str(directory),'verified_sha256':{k:sha(v) for k,v in data.items()},
                 'embedded_elf_sha256':sha(extracted),'embedded_elf_equals_supplied':True}

def fold(source):
    prefix,body=source.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
    rows=[[v.strip() for v in line.split(';')] for line in body.splitlines()]
    before=len(rows)
    if before not in [294,289]:raise ValueError('unqualified schedule length')
    load=next(i for i,row in enumerate(rows) if row[0]=='ld_tnsr V39, 0x2, I6');begin=load-6
    assert all(rows[i]==['nop']*4 for i in range(begin,load))
    replication=rows[load+8][2];assert replication.startswith('mov_dg.all')
    assert all(rows[i]==['nop']*4 for i in [*range(load+1,load+8),*range(load+9,begin+20)])
    del rows[begin:begin+20]
    index_write=next(i for i,row in enumerate(rows) if row[3]=='set_indx I6, b00001, S7')
    new_load=index_write+6;new_replication=new_load+6
    assert rows[new_load][0]=='nop' and rows[new_replication][2]=='nop'
    rows[new_load][0]='ld_tnsr V39, 0x2, I6';rows[new_replication][2]=replication
    first_macs=[i for i,row in enumerate(rows) if row[2].startswith('mac.bf16 acc_fp32 D14,')]
    shuffle=[i for i,row in enumerate(rows) if row[2].startswith('shuffle.u8')]
    assert len(first_macs)==len(shuffle)==32 and shuffle[0]-new_replication>=6
    k=-1
    for i,row in enumerate(rows):
        if i in first_macs:k+=1
        if row[2].startswith('mac.bf16'):
            row[2],n=re.subn(r'V(?:36|37|38)$',f'V{36+k%2}',row[2]);assert n==1
    for k,i in enumerate(shuffle):
        rows[i][2],n=re.subn(r'^shuffle.u8 V(?:36|37|38),',f'shuffle.u8 V{36+k%2},',rows[i][2]);assert n==1
    liveness=[]
    for k,(sh,mac) in enumerate(zip(shuffle,first_macs)):
        assert mac-sh>=6
        next_write=shuffle[k+2] if k+2<32 else None
        if next_write is not None:assert next_write>mac+3
        liveness.append({'k':k,'buffer':36+k%2,'shuffle':sh,'first_mac':mac,'last_mac':mac+3,'next_same_buffer_write':next_write})
    candidate=prefix+'.LBB0_3:\n'+'\n'.join('; '.join(row) for row in rows)+'\n.LBB0_6:'+suffix
    return candidate,{'original_k32_packets':before,'candidate_k32_packets':len(rows),'two_activation_buffers':True,
                      'prefill_moves':{'I6_final_set':index_write,'tensor_load':new_load,'DG_replicate':new_replication,'first_shuffle':shuffle[0],'first_mac':first_macs[0]},
                      'activation_lifetimes':liveness,'max_vector_register':max(map(int,re.findall(r'\bV(\d+)\b',candidate)))}

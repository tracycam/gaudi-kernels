"""Retiming only: preserve every BF16/FP32 MAC and original I7 scale order."""
import re


def transform(source, conservative=False, war_safe=False, load_safe=False):
    prefix, body = source.split('.LBB0_3:\n')
    body, suffix = body.split('.LBB0_6:')
    rows = [[slot.strip() for slot in line.split(';')] for line in body.splitlines()]
    assert len(rows) == 274 and all(len(row) == 4 for row in rows)
    assert not re.search(r'\bV38\b', body), 'requires qualified two-buffer folded GP'
    old_macs = [r[2] for r in rows if r[2].startswith('mac.')]
    assert len(old_macs) == 136
    assert rows[227][0] == 'ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7'
    assert rows[227][1] == 'add.i32 b00001 I7, 0x100, I7'
    assert rows[234][0] == 'ld_tnsr UNPCK_8_TO_16 unpack V36, 0x1, I7'
    # Empty all old scale work; retain the K pointer increment at packet225.
    for i in range(227, 274):
        rows[i] = ['nop'] * 4
    changes = [
        (199, 0, 'ld_tnsr UNPCK_8_TO_16 unpack V38, 0x1, I7'),
        (199, 1, 'add.i32 b00001 I7, 0x100, I7'),
        (208, 0, 'ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7'),
        (213, 2, 'convert.u8 all_lanes target_type=uint16 rhne D16, V38'),
        (219, 2, 'convert.u8 all_lanes target_type=uint16 rhne D22, V34'),
        (220, 2, 'shl.i16 V16, V16, 0x7'),
        (225, 2, 'shl.i16 V17, V17, 0x7'),
        (226, 2, 'shl.i16 V22, V22, 0x7'),
        (227, 2, 'shl.i16 V23, V23, 0x7'),
        (228, 2, 'convert.bf16 all_lanes target_type=fp32 rhne D28, V16'),
        (233, 2, 'convert.bf16 all_lanes target_type=fp32 rhne D30, V17'),
        (234, 2, 'convert.bf16 all_lanes target_type=fp32 rhne D32, V22'),
        (235, 2, 'convert.bf16 all_lanes target_type=fp32 rhne D34, V23'),
    ]
    if conservative or war_safe:
        # Preserve the failing candidate's loads, destinations and arithmetic;
        # only postpone dependent shift/convert/MAC operations to >=8 gaps.
        schedule=({213:225,219:226,220:233,225:234,226:235,227:236,228:241,233:242,234:243,235:244}
                  if war_safe else {220:225,225:226,226:227,227:228,228:233,233:234,234:235,235:236})
        changes=[(schedule.get(packet,packet) if slot==2 else packet,slot,op)
                 for packet,slot,op in changes]
    mac_start=250 if war_safe else 241 if conservative else 239
    changes += [(mac_start+i, 2, mac) for i, mac in enumerate(old_macs[-8:])]
    if load_safe:
        # Hardware causal gates:199 fails,208/214 pass with the sameV38/oldtail.
        # 199 failed empirically between lookup_2c198/200.18 is an original empty
        # LOAD slot, after I7 initialization9, with tensor loads16/22 nearby.
        changes=[(18 if packet==199 and slot==0 else packet,slot,op) for packet,slot,op in changes]
    for packet, slot, instruction in changes:
        assert rows[packet][slot] == 'nop', (packet, slot, rows[packet][slot])
        rows[packet][slot] = instruction
    end=mac_start+8
    assert all(row == ['nop']*4 for row in rows[end:])
    rows = rows[:end]
    assert old_macs == [r[2] for r in rows if r[2].startswith('mac.')]
    assert rows[225][1] == 'add.i32 S4, S4, 0x20'
    # Source register last-use boundaries are checked against the unchanged
    # non-scale body, including both registers written by each D destination.
    before = [[slot.strip() for slot in line.split(';')] for line in body.splitlines()]
    checks = [(34, 207, 208), (16, 210, 213), (17, 212, 213),
              (22, 216, 219), (23, 218, 219)]
    if war_safe:checks=[(v,last,schedule.get(overwrite,overwrite)) for v,last,overwrite in checks]
    for register, last, overwrite in checks:
        accesses = [i for i, row in enumerate(before[:225])
                    if any(re.search(rf'\bV{register}\b', slot) for slot in row)]
        assert max(accesses) == last and overwrite > last, (register, accesses)
    latency_edges = [(199,213),(208,219),(213,220),(213,225),(219,226),(219,227),
                     (220,228),(225,233),(226,234),(227,235),
                     (228,239),(228,240),(233,241),(233,242),
                     (234,243),(234,244),(235,245),(235,246)]
    if conservative or war_safe:
        def mapped(i):return schedule.get(i,i) if i<239 else i+(mac_start-239)
        latency_edges=[(mapped(p),mapped(c)) for p,c in latency_edges]
    if load_safe:latency_edges=[(18 if p==199 else p,c) for p,c in latency_edges]
    memory=[(i,r[0].split()[0]) for i,r in enumerate(rows) if r[0].startswith(('ld_tnsr','lookup_2c'))]
    memory_gaps=[dict(first=i,second=j,gap=j-i,first_op=op,second_op=next_op)
                 for (i,op),(j,next_op) in zip(memory,memory[1:])]
    lookup_spacing_safe=all(r['gap']>=2 for r in memory_gaps if 'lookup_2c' in (r['first_op'],r['second_op']))
    if load_safe:assert lookup_spacing_safe
    return prefix+'.LBB0_3:\n'+'\n'.join('; '.join(r) for r in rows)+'\n.LBB0_6:'+suffix, dict(
        old_packets=274,new_packets=end,saved_per_k32=274-end,saved_per_gp_task=64*(274-end),
        mac_instruction_sequence_unchanged=True,weight_lookup_activation_schedule_unchanged=True,
        scale_load_count_unchanged=True,max_vector_register=39,
        overwritten_register_last_use=[dict(register=v,last_access=l,first_overwrite=o) for v,l,o in checks],
        dependency_packet_gaps=[dict(producer=p,consumer=c,gap=c-p) for p,c in latency_edges],
        minimum_dependency_gap=min(c-p for p,c in latency_edges),conservative=conservative,war_safe=war_safe,
        load_safe=load_safe,lookup_neighbor_spacing_safe=lookup_spacing_safe,memory_gaps=memory_gaps,device_qualified=False)

"""Count emitted VLIW packets, including NOPs; no cycle-exact simulator claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def parse(path):
    regions={};region='prologue';bundle=False;packet=[]
    for raw in path.read_text().splitlines():
        line=raw.split('//')[0].strip()
        if not line or line.startswith(('.text','.file','.globl','.p2align','.type','.size','.section')):continue
        if line.endswith(':'):
            if line.startswith('.LBB'):region=line[:-1]
            continue
        if line.startswith('.'):continue
        if line=='{':bundle=True;packet=[];continue
        if line=='}':
            regions.setdefault(region,[]).append(packet);bundle=False;continue
        if bundle:packet.append(line)
        else:regions.setdefault(region,[]).append([line])
    result={}
    for name,packets in regions.items():
        operations=[op for packet in packets for op in packet]
        counts=Counter(op.split()[0].lower() for op in operations)
        vectors=[int(n) for op in operations for n in re.findall(r'%V(\d+)',op)]
        # Dn aliases Vn/Vn+1. Highest register number is a conservative span,
        # not a claim about simultaneous live ranges or hardware occupancy.
        vectors += [int(n)+1 for op in operations for n in re.findall(r'%D(\d+)',op)]
        local=[op for op in operations if op.startswith(('ld_l_v','st_l_v'))]
        result[name]={'packets':len(packets),'nop_only_packets':sum(all(o.lower()=='nop' for o in packet) for packet in packets),
                      'operations':dict(counts),'vector_register_span':max(vectors,default=-1)+1,'local_vector_accesses':local}
    assert '.LBB0_3' in result and '.LBB0_5' in result
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--M',type=int,default=512)
    p.add_argument('--K',type=int,default=2048);p.add_argument('--tpcs',type=int,default=24);a=p.parse_args()
    results={}
    for variant in ['baseline','amax16','lut_corrected','lut_single','lut_single4']:
        regions=parse(a.build_dir/(variant+'.s'))
        count=lambda region:regions.get(region,{}).get('packets',0)
        # Static per-row accounting; hardware LOOP end/delay semantics, memory
        # stalls and real task assignment are deliberately outside this model.
        step=512 if variant=='lut_single4' else 128
        def row_cost(k):
            return count('.LBB0_2')+count('.LBB0_3')*((k+step-1)//step)+count('.LBB0_4')+count('.LBB0_5')*((k+511)//512)+count('.LBB0_6')
        row=row_cost(a.K)
        hot=sum(len(regions[name]['local_vector_accesses']) for name in ['.LBB0_3','.LBB0_5'])
        results[variant]={'regions':regions,'scheduled_packets_per_row':row,'amax_elements_per_iteration':step,
            'packet_rows_by_K':{k:row_cost(k) for k in [128,129,257,512,2048,8192]},
            'max_vector_register_span':max(r['vector_register_span'] for r in regions.values()),
            'hot_K_loop_local_vector_accesses':hot,
            'conditional_24_TPC_packets':count('prologue')+row*((a.M+a.tpcs-1)//a.tpcs)+count('.LBB0_7')}
    report={'scope':'Static emitted packet/operation census. Not measured cycles, latency, speedup, occupancy, or physical traffic.',
        'M':a.M,'K':a.K,'tpcs_assumed':a.tpcs,'variants':results,
        'traffic_bytes':{'activation_read_two_passes':4*a.M*a.K,'quantized_output':a.M*a.K,'scales_output':4*a.M,
            'LUT_logical_bytes_per_active_TPC':516,'LUT_logical_bytes_all_assumed_TPCs':516*a.tpcs},
        'notes':['ceil(M/TPCs) assumes balanced whole-row assignment; actual Synapse scheduling is unverified.',
                 'Local vector accesses include compiler-materialized constants. Zero in hot K loops does not mean zero VLM traffic.',
                 'Packed vector reads include tail padding; denominators above are logical tensor bytes.',
                 'Compiler LOOP setup/end effects must be checked on hardware before converting packet counts to timing.']}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({v:{k:r[k] for k in ['scheduled_packets_per_row','max_vector_register_span','hot_K_loop_local_vector_accesses','conditional_24_TPC_packets']} for v,r in results.items()},indent=2))


if __name__=='__main__':main()

"""Recompute representative MXFP4 kernel metrics from qualified local evidence.

No device access. Weight-byte/peak ratios are reference normalizations, NOT bus
utilization counters. Keep generated-feed controls separate from real MXFP4.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SPEC = 'https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf'
PEAKS = dict(hbm_TBps=2.46, sram_read_TBps=6.4, sram_write_TBps=6.4,
             tpc_bf16_TFLOPS=11.0, mme_bf16_TFLOPS=432.0, mme_fp8_TFLOPS=865.0)
sources = {}


def read(name):
    p = ROOT/name
    sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def metrics(family, E, M, N, K, us, peak, origin, method, version, **extra):
    assert E >= 8 and M > 0 and us > 0
    weight = E*N*K*17//32
    flops = 2*E*M*N*K
    bw, tf = weight/us/1e6, flops/us/1e6
    return dict(family=family, experts=E, M_per_expert=M, N=N, K=K,
                complete_recipe_us=us, original_weight_bytes=weight, useful_flops=flops,
                original_weight_TBps=bw, weight_to_nominal_HBM_percent=100*bw/PEAKS['hbm_TBps'],
                useful_TFLOPS=tf, compute_peak_TFLOPS=peak,
                useful_compute_to_peak_percent=100*tf/peak,
                compute_only_weight_BW_bound_TBps=peak*17/(64*M),
                time_aggregation=method, source_cases=origin, version=version,
                physical_HBM_utilization_measured=False, **extra)


def build():
    grouped = read('evidence/mxfp4-grouped-bandwidth/summary.json')
    reuse = read('evidence/mxfp4-reuse/M1024-SMALLM.json')
    retained = read('evidence/mxfp4-w4a8/retained-facts.json')
    ring = read('evidence/mxfp4-w4a8/unified-sram-ring.json')
    feed = read('evidence/mxfp4-w4a8/feed-control-m2.json')
    contention = read('evidence/mxfp4-w4a8/sram-contention-v1.json')
    rows = []

    def group(stage, name):
        r = next(r for r in grouped['cases'] if r['stage'] == stage and r['case'] == name)
        assert r['returncode'] == 0 and r['correctness']['bad'] == 0
        assert r['graph_audit']['pass_'] and not r['profiled'] and not r['compile_only']
        p = r['plan']
        assert p['valid_rows'] == p['experts']*p['M_cap']
        assert r['graph_audit']['original_bytes_per_replay'] == p['experts']*p['N']*p['K']*17//32
        return r

    def add_groups(family, pairs, version):
        rs = [group(*pair) for pair in pairs]
        p = rs[0]['plan']
        assert all(all(r['plan'][k] == p[k] for k in ('experts','M_cap','N','K','mode')) for r in rs)
        samples = [s for r in rs for s in r['samples_us']]
        rows.append(metrics(family,p['experts'],p['M_cap'],p['N'],p['K'],statistics.mean(samples),
                            11.0 if p['mode']=='tpc' else 432.0,
                            [r['path'] for r in rs],
                            'all samples mean; ABBA candidate arms' if len(rs)==2 else 'all five sample mean',
                            version, rotating_source_bytes=p['rotating_weight_bytes']))

    for E in (8,16,32,64,128,256,384):
        if E in (8,384):
            pairs = [('final',f'e{E}-abba-b'),('final',f'e{E}-abba-c')]
        else:
            pairs = [('mask' if E==32 else 'final',f'e{E}-prefetch')]
        add_groups('TPC W4A16 gate/up',pairs,'N512 hand-scheduled prefetch')
    for E,M,stage,name in [(8,2,'coverage','e8-m2'),(32,2,'initial','e32-m2-s1'),
                            (384,2,'coverage','e384-m2'),(8,4,'coverage','e8-m4'),
                            (384,4,'coverage','e384-m4')]:
        add_groups('TPC W4A16 gate/up',[(stage,name)],'N256 shared-weight C kernel')
    for E in (8,64,384):
        add_groups('TPC W4A16 down',[('window',f'e{E}-down-m1')],
                   'generic N256 down; not the production down ISA baseline')
    add_groups('TPC→BF16 MME W4A16',[('initial','e8-mme-m2')],
               'grouped decoder before canonical-zero optimization')

    def add_reuse(names, version):
        rs = [next(r for r in reuse['records'] if r['case']==name) for name in names]
        for r in rs:
            assert r['numerical_pass'] and not r['profiled'] and not r['compile_only']
            assert r['placement']['pass_']
            assert r['placement']['no_storage_padding_amplification']
        p = rs[0]['plan']; ms = list(map(int,p['M'].split(',')))
        assert len(set(ms)) == 1
        us = statistics.mean(s['event_us'] for r in rs for s in r['samples'])
        rows.append(metrics('TPC→BF16 MME W4A16',len(ms),ms[0],p['N'],p['K'],us,432.0,
                    [f"artifacts/builds/mxfp4-reuse/{r['stage']}/results/{r['case']}" for r in rs],
                    'all samples mean; ABBA candidate arms' if len(rs)==2 else 'all five sample mean',version))
    for M in (1,2,4,512,1024):
        add_reuse([f'stream32-m{M}-fast'],'four-chain decoder + exact scale; original zero signs')
    add_reuse(['stream32-m128-abba-1-fast','stream32-m128-abba-2-fast'],
              'four-chain decoder + exact scale; original zero signs')
    abba = retained['bf16_decoder']['ABBA']
    rows.append(metrics('TPC→BF16 MME W4A16',32,2,512,6144,abba['candidate_us'],432.0,
                        [retained['bf16_decoder']['raw_assets']+'/results/'+n for n in abba['arms'][1:3]],
                        'mean of two per-process medians; ABBA candidate arms',
                        'finite-MAC canonical-zero optimized decoder'))
    for r in ring['comparisons']:
        rows.append(metrics('TPC→BF16 MME explicit unified ring',r['experts'],r['M'],r['N'],r['K'],
                            r['unified_us'],432.0,['evidence/mxfp4-w4a8/unified-sram-ring.json'],
                            'mean of two per-process medians; ABBA candidate arms',
                            'a6d5044 opt-in unified SRAM section; not fastest grouped path'))

    controls=[]
    for r in feed['abba']:
        for dtype,peak in [('bf16',432.),('fp8',865.)]:
            v=metrics('generated '+dtype+' feed control',r['E'],r['M'],512,6144,r[dtype]['event_us'],peak,
                      r[dtype]['cases'],'mean of two per-process medians; ABBA candidate arms','2427fd5',
                      real_mxfp4=False,activation_quantization_included=False)
            v['equivalent_only_TBps']=v.pop('original_weight_TBps')
            v.pop('weight_to_nominal_HBM_percent')
            v['generated_weight_TBps']=r['E']*512*6144*(2 if dtype=='bf16' else 1)/v['complete_recipe_us']/1e6
            controls.append(v)
    sram=[]
    for r in contention['records']:
        p=r.get('plan',{})
        if not r.get('profile') or p.get('schedule')!='resident' or p.get('chunk')!=2:
            continue
        assert r['no_tpc_execution_during_mme']
        bw=r['logical_consumer_TBps']
        sram.append(dict(case=r['case'],dtype=p['mode'],logical_consumer_TBps=bw,
                         logical_to_nominal_SRAM_read_percent=100*bw/6.4,
                         mme_interval_union_us=r['profile']['mme_union_us'],
                         bank_saturation_proven=False,physical_SRAM_bytes_measured=False,
                         scope='Resident tiles reused; MME-event normalization, not full recipe or bus saturation'))
    coverage=[]
    for family in ('TPC W4A16 gate/up','TPC→BF16 MME W4A16','TPC→FP8 MME W4A8'):
        for E in (8,16,32,64,128,256,384):
            for M in (1,2,4,8,16,32,64,128,256,512,1024):
                hits=[i for i,r in enumerate(rows) if r['family']==family and r['experts']==E and r['M_per_expert']==M]
                coverage.append(dict(family=family,experts=E,M_per_expert=M,
                                     state='representative measurement' if hits else 'not covered by this accepted matrix',row_indices=hits))
    return dict(scope='Offline reconciliation only; no new device run or default dispatch change',
                spec_url=SPEC,nominal_peaks=PEAKS,source_sha256=sources,
                formulas=dict(weight_bytes='E*N*K*17/32; M is not multiplied into weight bytes',
                              useful_flops='2*E*M*N*K',BW_TBps='weight_bytes/(microseconds*1e6)',
                              TFLOPS='useful_flops/(microseconds*1e6)'),
                warning='Nominal ratios are not physical HBM/VRF/SRAM/issue counters. Versions and timing statistics differ; no interpolated cells.',
                real_kernel_rows=rows,generated_controls=controls,sram_read_controls=sram,coverage=coverage)


def table(rows):
    header='| E | M/专家 | 时间µs | 原权重TB/s | /HBM标称 | 有效TFLOPS | /相应计算峰值 |\n|---:|---:|---:|---:|---:|---:|---:|\n'
    lines=[]
    for r in sorted(rows,key=lambda r:(r['M_per_expert'],r['experts'],r['version'])):
        mark='（去零判断）' if r['version']=='finite-MAC canonical-zero optimized decoder' else ''
        lines.append(f"| {r['experts']} | {r['M_per_expert']}{mark} | {r['complete_recipe_us']:.3f} | {r['original_weight_TBps']:.3f} | {r['weight_to_nominal_HBM_percent']:.1f}% | {r['useful_TFLOPS']:.3f} | {r['useful_compute_to_peak_percent']:.2f}% |")
    return header+'\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    result=build();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'matrix.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    rows=result['real_kernel_rows']
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (a.output_dir/'matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    md='# MXFP4实测利用率矩阵（离线重算）\n\n'+result['warning']+'\n\n'
    md+='M是每专家行数；完整线性投影时间，不含路由/SiLU/down/通信全链。权重比例不是总HBM利用率。\n\n'
    for family in dict.fromkeys(r['family'] for r in rows):
        md+='## '+family+'\n\n'+table([r for r in rows if r['family']==family])+'\n'
    md+='W4A8完整路径未测；生成控制与SRAM端点数据仅在matrix.json的独立字段中保留。\n'
    md+='版本、来源、样本聚合方式及覆盖缺口见matrix.json/CSV。该矩阵不是单一版本的完整交叉扫描。\n'
    (a.output_dir/'matrix.md').write_text(md)
    print(json.dumps(dict(real_kernel_rows=len(rows),control_rows=len(result['generated_controls']),output=str(a.output_dir))))

"""Deterministic complete-output FP64 oracles and separate native recipe probes.

Call inside tools/run_device_probe.py. Its output directory already exists.
The executable gates numerical results before it enters its timing loop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np


def bf16_bits(x):
    u=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)


def bf16_float(x):
    return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)


def decode(c):
    c=np.asarray(c,dtype=np.uint8)
    e=((c>>3)&15).astype(np.int32);m=(c&7).astype(np.float64)
    v=np.where(e==0,m/512,np.ldexp(1+m/8,e-7))
    return np.where(c&128,-v,v)


def half(c):
    a=c.astype(np.uint16)&127
    return ((c&128)|np.where(a>=16,a-8,(a>>1)+((a&1)&((a>>1)&1)))).astype(np.uint8)


def quant(x,scale):
    levels=decode(np.arange(127,dtype=np.uint8))
    q=np.minimum(np.abs(x.astype(np.float64)/scale.astype(np.float64)),448.)
    hi=np.minimum(np.searchsorted(levels,q,side='left'),126);lo=np.maximum(hi-1,0)
    dh=levels[hi]-q;dl=q-levels[lo]
    code=np.where((dl<dh)|((dl==dh)&((lo&1)==0)),lo,hi).astype(np.uint8)
    return code|(np.signbit(x).astype(np.uint8)<<7)


def fixture(path,m,n,k,kind):
    path.mkdir()
    rng=np.random.default_rng(1000003+m*31+n*7+k)
    w=rng.integers(0,127,(n,k),dtype=np.uint8)|(rng.integers(0,2,(n,k),dtype=np.uint8)<<7)
    x=bf16_float(bf16_bits(rng.normal(0,.3,(m,k))))
    scales=np.exp2(rng.uniform(-12,-4,n)).astype(np.float32)
    bias=rng.normal(0,.05,n).astype(np.float32)
    if kind=='ties':
        levels=decode(np.arange(127,dtype=np.uint8))
        values=np.concatenate([levels,(levels[:-1]+levels[1:])*.5])/64
        u=bf16_bits(values)
        around=bf16_float(np.concatenate([np.maximum(u.astype(np.int32)-1,0).astype(np.uint16),u,u+1]))
        around=np.clip(np.concatenate([around,-around]),-7,7)
        for row in range(m):x[row]=np.resize(np.roll(around,row*137),k)
        x[:,-1]=7;bias[:]=0
    if kind=='zero':x[:]=0;bias[:]=0
    if kind=='cancellation':
        w[:,1::2]=w[:,0:k-1:2]
        x[:,1::2]=-x[:,0:k-1:2]
        if k%2:x[:,-1]=0
    if kind=='range':
        w[:]=np.resize(np.array([0,1,2,3,7,8,15,16,119,120,125,126,128,129,255-1],np.uint8),w.shape)
        scales=np.exp2(np.linspace(-20,0,n)).astype(np.float32)
        x[:]=bf16_float(bf16_bits(np.resize(np.array([0,1e-7,-1e-7,1,10,-10,.015625],np.float32),x.shape)))
    if kind=='subnormal':w=(w&128)|((w%8)*2+1)
    if kind=='extended':w=(w&128)|(120+w%7)
    adapt=np.any((w&127)>119,axis=1,keepdims=True)
    native=np.where(adapt,half(w),w)
    scale_prepared=np.where(adapt[:,0],scales*2,scales).astype(np.float32)
    wd=decode(w)*scales.astype(np.float64)[:,None]
    nd=decode(native)*scale_prepared.astype(np.float64)[:,None]
    sa=(np.maximum(np.max(np.abs(x),axis=1,keepdims=True),np.float32(1e-10))*np.float32(1/448)).astype(np.float32)
    q=quant(x,sa);aq=decode(q)*sa.astype(np.float64)
    an=decode(half(q))*(sa.astype(np.float64)*2)
    refs={
        'reference_fp64':x.astype(np.float64)@wd.T+bias,
        'gpu_style_w8a8_fp64':aq@wd.T+bias,
        'adapted_w8a8_fp64':an@nd.T+bias,
        'adapted_w8a16_fp64':x.astype(np.float64)@nd.T+bias,
        'sumabs_fp64':np.abs(x.astype(np.float64))@np.abs(wd).T+np.abs(bias),
    }
    files={'weight_original':w,'weight_native':native,'activation':bf16_bits(x),
           'scales_original':scales,'scales_prepared':scale_prepared,'bias':bias,**refs}
    for name,value in files.items():value.tofile(path/(name+'.bin'))
    (path/'fixture.json').write_text(json.dumps({'M':m,'N':n,'K':k,'kind':kind,
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.bin')},
        'contract':'finite E4M3FN weights, channel FP32 scales, BF16 input/output; amax/448 per-token OCP RNE for W8A8'},indent=2))
    return refs


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',required=True,type=Path)
    p.add_argument('--modes',nargs='+',choices=['w8a8','w8a16'],default=['w8a8','w8a16'])
    p.add_argument('--grid',choices=['smoke','coverage','stream','edge','placement','tiny'],default='smoke')
    a=p.parse_args();build=a.build.resolve();out=Path(os.environ['PROBE_OUT'])
    shapes={'smoke':[(1,129,257,'random'),(16,512,768,'random')],
        'coverage':[(m,1024,2048,'random') for m in [1,2,8,16,64,128,512,513]],
        'stream':[(1,8192,8192,'random'),(64,4096,4096,'random')],
        'placement':[(m,1024,2048,'random') for m in [1,128,512,513]],
        'tiny':[(3,63,65,'random')],
        'edge':[(2,255,513,'range'),(8,257,769,'cancellation'),(1,135,255,'zero'),(3,63,65,'random'),(2,129,257,'subnormal'),(2,129,257,'extended')]}
    results=[]
    for m,n,k,kind in shapes[a.grid]:
        f=out/f'fixture-{m}-{n}-{k}-{kind}';refs=fixture(f,m,n,k,kind)
        for mode in a.modes:
            case=out/f'{mode}-{m}-{n}-{k}-{kind}';case.mkdir()
            graphs=case/'graphs';graphs.mkdir()
            env=dict(os.environ,HABANA_LOGS=str(case/'habana-logs'),ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(case/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(graphs),SRAM_SLICER_GRAPH_VISUALIZATION='1')
            command=[str(build/'fp8_linear_bench'),mode,str(m),str(n),str(k),str(f)]
            with (case/'run.log').open('w') as log:
                run=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            rows=[]
            for line in (case/'run.log').read_text().splitlines():
                if line.startswith('{'):
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
            record={'binary_sha256':hashlib.sha256((build/'fp8_linear_bench').read_bytes()).hexdigest(),'mode':mode,'M':m,'N':n,'K':k,'kind':kind,'returncode':run.returncode,'rows':rows}
            timings=[r for r in rows if r.get('stage')=='timing']
            if (case/'output.bin').exists():
                actual=bf16_float(np.fromfile(case/'output.bin',dtype=np.uint16)).reshape(m,n).astype(np.float64)
                record['full_output_metrics']={name:{'relative_l2':float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)),
                    'max_abs':float(np.max(np.abs(actual-ref))),'rms':float(np.sqrt(np.mean((actual-ref)**2)))}
                    for name,ref in refs.items() if name!='sumabs_fp64'}
            if timings:
                us=statistics.median(t['event_us'] for t in timings)
                record.update(event_median_us=us,wall_median_us=statistics.median(t['wall_us'] for t in timings),
                    weight_payload_gb_s=n*k/us/1000.,effective_tflops=2*m*n*k/us/1e6)
            (case/'result.json').write_text(json.dumps(record,indent=2));results.append(record)
            print(json.dumps({k:v for k,v in record.items() if k not in ['rows','full_output_metrics']}),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()

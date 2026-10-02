"""Lossless checkpoint-row-major <-> shared TPC/MME storage conversion.

Input nibble convention: K-even in low nibble, K-odd in high nibble.
Call on an aligned matrix or expert shard. Only one 512-row tile is expanded
to byte codes on the CPU at a time; no BF16 weight copy is constructed.
"""
import numpy as np

def pack(packed_rows, scales):
    p=np.asarray(packed_rows);s=np.asarray(scales)
    if p.dtype!=np.uint8 or s.dtype!=np.uint8 or p.ndim!=2:raise ValueError('uint8 2D arrays required')
    N,K2=p.shape;K=K2*2
    if N%512 or K%32 or s.shape!=(N,K//32):raise ValueError('N%512 == K%32 == 0 and per-32-K scales required')
    w=np.empty((N//512,K,256),np.uint8);out_s=np.empty((N//512,K//32,512),np.uint8)
    for b in range(N//512):
        src=p[b*512:(b+1)*512];codes=np.empty((512,K),np.uint8)
        codes[:,0::2]=src&15;codes[:,1::2]=src>>4;c=codes.T
        for q in (0,1):w[b,:,q*128:(q+1)*128]=c[:,q*256:q*256+128]|(c[:,q*256+128:q*256+256]<<4)
        out_s[b]=s[b*512:(b+1)*512].T
    return w,out_s

def unpack(blocked, scales):
    w=np.asarray(blocked);s=np.asarray(scales)
    if w.dtype!=np.uint8 or s.dtype!=np.uint8 or w.ndim!=3 or w.shape[2]!=256:raise ValueError('uint8 [N/512,K,256] required')
    B,K,_=w.shape
    if K%32 or s.shape!=(B,K//32,512):raise ValueError('bad scales')
    out=np.empty((B*512,K//2),np.uint8);out_s=np.empty((B*512,K//32),np.uint8)
    for b in range(B):
        c=np.empty((K,512),np.uint8)
        for q in (0,1):c[:,q*256:q*256+128]=w[b,:,q*128:(q+1)*128]&15;c[:,q*256+128:q*256+256]=w[b,:,q*128:(q+1)*128]>>4
        out[b*512:(b+1)*512]=(c[0::2]|(c[1::2]<<4)).T;out_s[b*512:(b+1)*512]=s[b].T
    return out,out_s

if __name__=='__main__':
    rng=np.random.default_rng(1847)
    for N,K in ((512,128),(1024,256),(4096,6144)):
        p=rng.integers(0,256,(N,K//2),dtype=np.uint8);s=rng.integers(0,256,(N,K//32),dtype=np.uint8)
        w,t=pack(p,s);p2,s2=unpack(w,t)
        assert w.nbytes+t.nbytes==p.nbytes+s.nbytes
        assert np.array_equal(p,p2) and np.array_equal(s,s2)
        print('BIT-EXACT PACKING',N,K,p.nbytes+s.nbytes)

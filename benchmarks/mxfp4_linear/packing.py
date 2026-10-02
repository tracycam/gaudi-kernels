"""Layout v1, lossless E2M1 byte rearrangement with explicit scale validation.

Checkpoint rows use K-even in the low nibble, K-odd in the high nibble.
Odd logical K is accepted only when the unused checkpoint nibble is zero.
Only bounded 256-row code tiles are unpacked; no BF16 weight cache is created.
"""
import numpy as np

def pack(rows, scales, logical_k=None):
    rows=np.asarray(rows); scales=np.asarray(scales)
    if rows.dtype!=np.uint8 or scales.dtype!=np.uint8 or rows.ndim!=2:
        raise ValueError('uint8 checkpoint rows and E8M0 scales required')
    n,k2=rows.shape; k=logical_k if logical_k is not None else k2*2
    if n<1 or k<1 or k2!=(k+1)//2 or scales.shape!=(n,(k+31)//32):
        raise ValueError('incompatible checkpoint/scales/logical K')
    if np.any((scales<2)|(scales>252)):
        raise ValueError('exact normal BF16 path requires E8M0 codes 2..252; 255 is NaN')
    if k%2 and np.any(rows[:,-1]&240):
        raise ValueError('odd-K unused high nibble must be zero')
    blocks=(n+255)//256
    packed=np.zeros((blocks,k,128),np.uint8)
    out_scales=np.full((blocks,(k+31)//32,256),127,np.uint8)
    for b in range(blocks):
        count=min(256,n-b*256); r=rows[b*256:b*256+count]
        codes=np.zeros((256,k2*2),np.uint8)
        codes[:count,0::2]=r&15; codes[:count,1::2]=r>>4
        packed[b]=codes[:128,:k].T|(codes[128:,:k].T<<4)
        out_scales[b,:,:count]=scales[b*256:b*256+count].T
    return packed,out_scales

def unpack(packed,scales,logical_n):
    packed=np.asarray(packed); scales=np.asarray(scales)
    if packed.dtype!=np.uint8 or scales.dtype!=np.uint8 or packed.ndim!=3 or packed.shape[2]!=128:
        raise ValueError('uint8 layout-v1 tensors required')
    blocks,k,_=packed.shape;n=logical_n
    if n<1 or blocks!=(n+255)//256 or scales.shape!=(blocks,(k+31)//32,256):
        raise ValueError('incompatible layout-v1 dimensions')
    rows=np.zeros((n,(k+1)//2),np.uint8); out_scales=np.empty((n,(k+31)//32),np.uint8)
    for b in range(blocks):
        count=min(256,n-b*256)
        codes=np.concatenate((packed[b]&15,packed[b]>>4),axis=1).T[:count]
        rows[b*256:b*256+count]=codes[:,0::2]
        rows[b*256:b*256+count,:k//2]|=codes[:,1::2]<<4
        out_scales[b*256:b*256+count]=scales[b,:,:count].T
    return rows,out_scales

if __name__=='__main__':
    rng=np.random.default_rng(1907)
    for n,k in [(1,1),(129,37),(256,128),(513,257),(512,6144),(6144,256)]:
        rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8)
        if k%2:rows[:,-1]&=15
        scales=rng.integers(2,253,(n,(k+31)//32),dtype=np.uint8)
        p,s=pack(rows,scales,k);r,t=unpack(p,s,n)
        assert np.array_equal(r,rows) and np.array_equal(t,scales)
        print('PASS lossless',n,k,'logical bytes',rows.nbytes+scales.nbytes,'padded bytes',p.nbytes+s.nbytes)
    for exponent in [0,1,253,254,255]:
        try:pack(np.zeros((1,16),np.uint8),np.full((1,1),exponent,np.uint8))
        except ValueError:pass
        else:raise AssertionError('unsupported scale accepted')
    print('PASS unsupported scale rejection')

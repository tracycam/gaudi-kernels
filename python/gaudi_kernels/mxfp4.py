"""Experimental MXFP4 N256 layout-v1 and host-known static grouped graph nodes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedMXFP4:
    weight: object
    scales: object
    n: int
    k: int
    layout_version: int = 1

    def to(self, device):
        if self.layout_version!=1:raise ValueError('unsupported prepared MXFP4 version')
        return PreparedMXFP4(self.weight.to(device),self.scales.to(device),self.n,self.k,self.layout_version)


def prepare_mxfp4(rows,scales,*,logical_k=None):
    """Lossless CPU load-time packing; accepted E8M0 codes2..252 only."""
    import torch
    if rows.device.type!='cpu' or scales.device.type!='cpu' or rows.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or rows.ndim!=2:
        raise ValueError('CPU uint8 checkpoint rows/scales required')
    n,k2=rows.shape;k=2*k2 if logical_k is None else logical_k
    if not isinstance(k,int) or n<1 or k<1 or k2!=(k+1)//2 or tuple(scales.shape)!=(n,(k+31)//32):
        raise ValueError('MXFP4 row/scales/logical K shape mismatch')
    if bool(((scales<2)|(scales>252)).any()):raise ValueError('exact normal BF16 route requires E8M0 codes2..252')
    if k%2 and bool((rows[:,-1]&240).any()):raise ValueError('unused odd-K nibble must be zero')
    blocks=(n+255)//256;packed=torch.zeros((blocks,k,128),dtype=torch.uint8)
    exponents=torch.full((blocks,(k+31)//32,256),127,dtype=torch.uint8)
    for b in range(blocks):
        count=min(256,n-b*256);src=rows[b*256:b*256+count]
        codes=torch.zeros((256,k2*2),dtype=torch.uint8)
        codes[:count,0::2]=src&15;codes[:count,1::2]=src>>4
        packed[b]=codes[:128,:k].T|(codes[128:,:k].T<<4)
        exponents[b,:,:count]=scales[b*256:b*256+count].T
    return PreparedMXFP4(packed,exponents,n,k)


def make_mxfp4_lut():
    """One shared CPU BF16[1,512] byte-key table, transfer once at load time."""
    import torch
    values=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.bfloat16)
    keys=torch.arange(256)
    return torch.stack((values[keys&15],values[keys>>4]),dim=1).reshape(1,512)


def linear_mxfp4(x,prepared,lut,bias=None):
    """Original BF16 activation -> hidden SRAM decode/MME -> FP32 output.

    SRAM placement is graph/shape dependent and must be audited. No decoded
    weight escapes this function. Caller keeps prepared tensors and LUT alive.
    """
    import torch
    if not isinstance(prepared,PreparedMXFP4) or prepared.layout_version!=1:
        raise ValueError('MXFP4 N256 layout-v1 prepared weight required')
    decoded=torch.ops.gaudi_kernels._mxfp4_decode(prepared.weight,prepared.scales,lut,prepared.n)
    y=torch.ops.gaudi_kernels._mxfp4_mm(x,decoded)
    return y if bias is None else torch.ops.gaudi_kernels._mxfp4_bias(y,bias)


def grouped_mxfp4_static(inputs,weights,lut,biases=None):
    """Host-known groups in one current graph; empty groups return None.

    Shape changes require a new plan/capture. This does not implement device
    routing/counts and must not be used to read per-layer routing back to CPU.
    """
    if biases is None:biases=[None]*len(inputs)
    if len(inputs)!=len(weights) or len(inputs)!=len(biases):raise ValueError('group lengths differ')
    return tuple(None if x.shape[0]==0 else linear_mxfp4(x,w,lut,b) for x,w,b in zip(inputs,weights,biases))

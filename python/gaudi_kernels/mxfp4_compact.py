"""MXFP4 compact layout-v3: original nibble/E8M0 bytes, no N/K tile padding.

The full N512/K32 body is byte-identical to native-v2 on aligned shapes.
Ragged N and K regions retain checkpoint row packing. Packing is load-time CPU
work only. This is a storage contract, not a claim that floating-point MACs
preserve FP32 subnormal contributions; an arithmetic guard/repair remains needed.
"""
from dataclasses import dataclass
import struct
import numpy as np

LAYOUT_VERSION = 3
HEADER = struct.Struct('<8sIIIIII')  # 32 host metadata bytes; not weight/scale payload
MAGIC = b'MX4CP3\0\0'


@dataclass(frozen=True)
class Segment:
    kind: str
    n_begin: int
    n_count: int
    k_begin: int
    k_count: int
    weight_begin: int
    weight_end: int
    scale_begin: int
    scale_end: int


@dataclass(frozen=True)
class PreparedCompact:
    weight: np.ndarray  # flat uint8; exactly N*ceil(K/2) bytes
    scales: np.ndarray  # flat uint8; exactly N*ceil(K/32) bytes
    n: int
    k: int
    layout_version: int = LAYOUT_VERSION

    def validate(self):
        if (self.layout_version != LAYOUT_VERSION
                or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
                       for v in (self.n, self.k)) or self.n < 1 or self.k < 1):
            raise ValueError('compact MXFP4 layout-v3 and positive N/K required')
        if max(self.n, self.k, self.n*((self.k+1)//2),
               self.n*((self.k+31)//32)) > 2147483647:
            raise ValueError('compact TPC tensor dimensions and byte offsets must fit int32')
        for array, size in ((self.weight, self.n*((self.k+1)//2)),
                            (self.scales, self.n*((self.k+31)//32))):
            if (not isinstance(array, np.ndarray) or array.dtype != np.uint8
                    or array.shape != (size,) or not array.flags.c_contiguous):
                raise ValueError('compact payload must be contiguous flat uint8 with exact size')
        if np.any(self.scales == 255):
            raise ValueError('E8M0 code255 is NaN; finite codes0..254 are supported')
        return self

    @property
    def n_full(self):
        return self.n//512*512

    @property
    def k_full(self):
        return self.k//32*32

    @property
    def payload_bytes(self):
        return self.weight.nbytes + self.scales.nbytes

    @property
    def bf16_fast_decode_eligible(self):
        """Load-time arithmetic-route metadata; extreme scales need exact repair."""
        return bool(np.all((self.scales >= 2) & (self.scales <= 252)))

    @property
    def aligned_native_v2(self):
        return self.n % 512 == 0 and self.k % 32 == 0

    def metadata(self):
        """Fixed 32-byte host header; segment descriptors are derived, not uploaded."""
        self.validate()
        return HEADER.pack(MAGIC, self.layout_version, self.n, self.k,
                           self.n_full, self.k_full, int(self.bf16_fast_decode_eligible))

    def as_native_v2(self):
        """Zero-copy compatibility for aligned, fast-scale payloads only.

        This storage view adds no runtime work or padding. The v2 arithmetic
        backend still needs its own activation/FP32-subnormal correctness gate.
        """
        self.validate()
        if not self.aligned_native_v2 or not self.bf16_fast_decode_eligible:
            raise ValueError('native-v2 view needs N512/K32 alignment and scales2..252')
        from .mxfp4_prepared_gemv import PreparedN512
        return PreparedN512(self.weight.reshape(-1, 256), self.scales.reshape(-1, 512),
                            self.n, self.k)

    def segments(self):
        nf, kf, g = self.n_full, self.k_full, self.k_full//32
        body_w, body_s = nf*kf//2, nf*g
        prefix_w, prefix_s = self.n*kf//2, self.n*g
        regions = []
        if nf and kf:
            regions.append(Segment('native_n512', 0, nf, 0, kf,
                                   0, body_w, 0, body_s))
        if self.n > nf and kf:
            regions.append(Segment('n_tail_rows', nf, self.n-nf, 0, kf,
                                   body_w, prefix_w, body_s, prefix_s))
        if self.k > kf:
            regions.append(Segment('k_tail_rows', 0, self.n, kf, self.k-kf,
                                   prefix_w, self.weight.size, prefix_s, self.scales.size))
        return tuple(regions)

    def code_address(self, n, k):
        """O(1) (byte offset, nibble shift), shared with the exact/repair reader."""
        if not 0 <= n < self.n or not 0 <= k < self.k:
            raise IndexError('logical MXFP4 element outside N/K')
        nf, kf = self.n_full, self.k_full
        if k >= kf:
            return self.n*kf//2 + n*((self.k-kf+1)//2) + (k-kf)//2, 4*((k-kf)%2)
        if n >= nf:
            return nf*kf//2 + (n-nf)*(kf//2) + k//2, 4*(k%2)
        p = (n%512//128)*128 + 2*(n%64) + (n%128)//64
        return (n//512)*kf*256 + k*256 + (p//256)*128 + p%128, 4*((p%256)//128)

    def scale_address(self, n, group):
        if not 0 <= n < self.n or not 0 <= group < (self.k+31)//32:
            raise IndexError('logical MXFP4 scale outside N/K32')
        nf, kf, g = self.n_full, self.k_full, self.k_full//32
        if group >= g:
            return self.n*g+n
        if n >= nf:
            return nf*g+(n-nf)*g+group
        p = (n%512//128)*128 + 2*(n%64) + (n%128)//64
        return (n//512)*g*512+group*512+p

    def get_code(self, n, k):
        address, shift = self.code_address(n, k)
        return (int(self.weight[address]) >> shift) & 15

    def get_scale(self, n, group):
        return int(self.scales[self.scale_address(n, group)])


def prepare(rows, scales, *, logical_k=None):
    """Lossless CPU-only packing, including an unused odd-K high nibble.

    Unlike padded native-v2, all checkpoint bytes appear exactly once. No
    activation, M-dependent representation, or expanded weight is prepared.
    """
    if (not isinstance(rows, np.ndarray) or not isinstance(scales, np.ndarray)
            or rows.dtype != np.uint8 or scales.dtype != np.uint8 or rows.ndim != 2):
        raise ValueError('CPU numpy uint8 checkpoint rows/scales required')
    n, width = rows.shape
    k = width*2 if logical_k is None else logical_k
    if (not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_))
            or n < 1 or k < 1 or width != (k+1)//2 or scales.shape != (n, (k+31)//32)):
        raise ValueError('checkpoint row/scale shape or logical K mismatch')
    k = int(k)
    if np.any(scales == 255):
        raise ValueError('E8M0 code255 is NaN; finite codes0..254 are supported')
    nf, kf = n//512*512, k//32*32
    g = kf//32
    weight = np.empty(n*((k+1)//2), np.uint8)
    exponents = np.empty(n*((k+31)//32), np.uint8)
    lanes = np.arange(512)
    permutation = (lanes//128)*128+(lanes%128)//2+(lanes%2)*64
    for block in range(nf//512):
        codes = np.empty((512, kf), np.uint8)
        src = rows[block*512:(block+1)*512, :kf//2]
        codes[:, ::2], codes[:, 1::2] = src & 15, src >> 4
        codes = codes[permutation]
        packed = weight[block*kf*256:(block+1)*kf*256].reshape(kf, 256)
        for pair in range(2):
            packed[:, pair*128:(pair+1)*128] = (codes[pair*256:pair*256+128].T
                | (codes[pair*256+128:(pair+1)*256].T << 4))
        exponents[block*g*512:(block+1)*g*512] = scales[block*512:(block+1)*512, :g][permutation].T.ravel()
    weight[nf*kf//2:n*kf//2] = rows[nf:, :kf//2].ravel()
    exponents[nf*g:n*g] = scales[nf:, :g].ravel()
    if k > kf:
        weight[n*kf//2:] = rows[:, kf//2:].ravel()
        exponents[n*g:] = scales[:, g:].ravel()
    return PreparedCompact(weight, exponents, n, k).validate()


def unpack(prepared):
    """Byte-exact inverse; preserves both signed-zero codes and unused nibbles."""
    p = prepared.validate()
    rows = np.zeros((p.n, (p.k+1)//2), np.uint8)
    scales = np.empty((p.n, (p.k+31)//32), np.uint8)
    for n in range(p.n):
        for k in range(p.k):
            rows[n, k//2] |= p.get_code(n, k) << (4*(k%2))
        for g in range((p.k+31)//32):
            scales[n, g] = p.get_scale(n, g)
        if p.k % 2:
            address, _ = p.code_address(n, p.k-1)
            rows[n, -1] |= p.weight[address] & 240
    return rows, scales


def from_buffers(weight, scales, metadata):
    """Deserialize without silently repairing version, dimensions, or payload size."""
    if len(metadata) != HEADER.size:
        raise ValueError('compact header must contain exactly 32 bytes')
    magic, version, n, k, nf, kf, flags = HEADER.unpack(metadata)
    if (magic != MAGIC or version != LAYOUT_VERSION or flags not in (0, 1)
            or nf != n//512*512 or kf != k//32*32):
        raise ValueError('incompatible compact MXFP4 metadata')
    result = PreparedCompact(np.asarray(weight), np.asarray(scales), n, k, version).validate()
    if flags != int(result.bf16_fast_decode_eligible):
        raise ValueError('fast-decode metadata disagrees with original E8M0 bytes')
    return result

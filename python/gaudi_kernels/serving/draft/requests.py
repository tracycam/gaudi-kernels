# SPDX-License-Identifier: Apache-2.0
"""Request-owned committed drafter context; target KV remains scheduler-owned."""
import torch
from torch import nn

from gaudi_kernels.serving.draft.context import CommittedContext


class ContextPool(nn.Module):
    """Stable device storage with host ownership and device row selection.

    Rejected/padded queries write only the per-slot sink. Releasing a request
    does not touch in-flight device work; acquiring its slot resets visibility
    before the new prompt is committed, in the worker's ordered stream.
    """
    def __init__(self, draft, capacity, device):
        super().__init__()
        spec = draft.spec
        self.storage = CommittedContext(layers=spec.layers, batch=capacity, window=spec.window,
                                        kv_heads=spec.kv_heads, dim=spec.dim, device=device)
        self.owners = {}
        self.free = list(reversed(range(capacity)))
        self.device = device
        window = self.storage.window
        self._layers = spec.layers
        self._pow2_mask = (window - 1) if (window & (window - 1) == 0) else None
        dtype = getattr(self.storage, 'key_0').dtype
        self.register_buffer(
            '_kv_all',
            torch.zeros(capacity, window + 1, 2 * spec.layers, spec.kv_heads, spec.dim,
                        device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            '_arange_window',
            torch.arange(window, dtype=torch.int32, device=device)[None],
            persistent=False,
        )
        self._bind_views()

    def _bind_views(self):
        window = self.storage.window
        for layer in range(self._layers):
            self.storage._buffers[f'key_{layer}'] = self._kv_all[:, :, 2 * layer]
            self.storage._buffers[f'value_{layer}'] = self._kv_all[:, :, 2 * layer + 1]
        self._kv_flat = self._kv_all.view(-1, 2 * self._layers, self._kv_all.shape[3], self._kv_all.shape[4])
        self._kv_win = self._kv_all[:, :window]
        self._pos_flat = self.storage.positions.view(-1)
        self._pos_win = self.storage.positions[:, :window]
        self.device = self._kv_all.device
        self._slot_masks = tuple(
            torch.arange(self.storage.batch, device=self.device) == slot
            for slot in range(self.storage.batch)
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._bind_views()
        return self

    def release(self, request_ids):
        for request_id in request_ids:
            slot = self.owners.pop(request_id, None)
            if slot is not None:
                self.free.append(slot)

    def slot(self, request_id):
        slot = self.owners.get(request_id)
        if slot is None:
            if not self.free:
                raise RuntimeError('Drafter context pool exhausted; request ownership was not released')
            slot = self.free.pop()
            self.owners[request_id] = slot
            self.storage.reset(self._slot_masks[slot])
            self._kv_all[slot].zero_()
        return slot

    def commit_slots(self, projected, positions, counts, slots):
        window = self.storage.window
        b, t = positions.shape
        if len(projected) != self._layers or t > window or counts.shape != (b,) or slots.shape != (b,):
            raise ValueError('Invalid pooled context commit layout')
        arange_t = (self._arange_window[:, :t]
                    if self._arange_window.device == positions.device
                    else torch.arange(t, dtype=torch.int32, device=positions.device)[None])
        accepted = (arange_t < counts[:, None]) & (positions >= 0)
        mod_pos = (positions & self._pow2_mask) if self._pow2_mask is not None else (positions % window)
        offsets = torch.where(accepted, mod_pos, window)
        indices = (slots[:, None].long() * (window + 1) + offsets.long()).reshape(-1)
        flat_pairs = []
        for k_val, v_val in projected:
            flat_pairs.append(k_val)
            flat_pairs.append(v_val)
        kv_stacked = torch.stack(flat_pairs, dim=2)
        kv_masked = torch.where(accepted[:, :, None, None, None], kv_stacked, 0).reshape(-1, *self._kv_flat.shape[1:])
        self._kv_flat.index_copy_(0, indices, kv_masked)
        self._pos_flat.index_copy_(
            0, indices, torch.where(accepted, positions, -1).int().reshape(-1))

    def read(self, slots):
        indices = slots.long()
        kv_sel = self._kv_win.index_select(0, indices)
        pos = self._pos_win.index_select(0, indices)
        return (tuple((kv_sel[:, :, 2 * i], kv_sel[:, :, 2 * i + 1]) for i in range(self._layers)),
                pos, pos >= 0)


def prompt_slices(positions, query_counts, *, merged):
    """Describe valid rows in HPU packed or padded prompt tensors, on the host."""
    if positions.ndim != 2 or any(count < 0 for count in query_counts):
        raise ValueError('Invalid prompt layout')
    if merged:
        if positions.shape[0] != 1 or sum(query_counts) > positions.shape[1]:
            raise ValueError('Invalid merged prompt layout')
        offset = 0
        result = []
        for count in query_counts:
            result.append(slice(offset, offset + count))
            offset += count
        return result
    if len(query_counts) > positions.shape[0] or max(query_counts, default=0) > positions.shape[1]:
        raise ValueError('Invalid padded prompt layout')
    stride = positions.shape[1]
    return [slice(i * stride, i * stride + count) for i, count in enumerate(query_counts)]

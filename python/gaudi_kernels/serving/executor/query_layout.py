"""Declared decode/verify query ownership, independent of token/position values.

The qualified plugin pads every request to max(query_lengths), then flattens
request-major. This describes that source layout; it does not admit native
T>1 execution, infer ownership from IDs, or allocate/commit KV.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeQueries:
    request_ids: tuple
    lengths: tuple
    positions: tuple
    padded_rows: int
    speculative: bool

    def __post_init__(self):
        n=len(self.request_ids)
        if (not n or len(set(self.request_ids))!=n or len(self.lengths)!=n or len(self.positions)!=n or
                any(not isinstance(r,str) or not r for r in self.request_ids) or
                any(type(v) is not int or v<1 for v in self.lengths) or
                any(type(v) is not int or v<0 for v in self.positions) or
                type(self.padded_rows) is not int or self.padded_rows<n*self.capacity or
                self.padded_rows%self.capacity or type(self.speculative) is not bool):
            raise ValueError('Invalid declared decode query layout')

    @property
    def capacity(self):
        return max(self.lengths)

    @property
    def active_rows(self):
        return tuple(i*self.capacity+j for i,n in enumerate(self.lengths) for j in range(n))

    def validate(self, positions, active_rows, selected):
        """Return request IDs for each selected output row, None for padding."""
        if len(positions)!=self.padded_rows or tuple(sorted(active_rows))!=self.active_rows:
            raise ValueError('Attention query visibility differs from declared requests')
        for i,(n,start) in enumerate(zip(self.lengths,self.positions)):
            if positions[i*self.capacity:i*self.capacity+n]!=list(range(start,start+n)):
                raise ValueError('Query positions differ from declared request cursor')
        expected_rows=self.padded_rows if self.speculative else self.padded_rows//self.capacity
        if len(selected)!=expected_rows:
            raise ValueError('Selected logits shape differs from declared query layout')
        owners=[]
        if self.speculative:
            active=set(self.active_rows)
            for row,query in enumerate(selected):
                if row not in active:
                    owners.append(None)
                elif query!=row:
                    raise ValueError('Speculative logits select another request/query')
                else:
                    owners.append(self.request_ids[row//self.capacity])
        else:
            for i,query in enumerate(selected):
                if i>=len(self.request_ids):
                    owners.append(None)
                elif query!=i*self.capacity+self.lengths[i]-1:
                    raise ValueError('Decode logits do not select the declared final query')
                else:
                    owners.append(self.request_ids[i])
        return owners

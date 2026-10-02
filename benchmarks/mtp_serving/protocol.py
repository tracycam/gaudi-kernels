"""Greedy device acceptance and CPU KV transaction reference.

The anchor input is already emitted but not yet in committed KV. A cycle
verifies [anchor, drafts], emits [accepted drafts, target correction/bonus],
and commits the same number of input rows. Last emitted token is next anchor.
This implements greedy only; stochastic acceptance is deliberately separate.
"""
def greedy_accept(drafts,target_ids,draft_lengths,remaining,eos_id):
    import torch
    b,k=drafts.shape
    assert target_ids.shape==(b,k+1)and draft_lengths.shape==remaining.shape==(b,)
    columns=torch.arange(k+1,device=drafts.device,dtype=torch.int32).unsqueeze(0)
    match=(drafts==target_ids[:,:k])&(columns[:,:k]<draft_lengths[:,None])
    accepted=match.to(torch.int32).cumprod(1,dtype=torch.int32).sum(1,dtype=torch.int32)
    # At rejection, target_ids[accepted] is the correction; after all accepts it
    # is the bonus. No target logits/probabilities are silently approximated.
    choices=torch.cat((drafts,target_ids[:,-1:]),1)
    choices=torch.where(columns==accepted[:,None],target_ids,choices)
    advance=torch.minimum(accepted+1,remaining)
    eos=torch.where((choices==eos_id)&(columns<advance[:,None]),columns+1,k+2).amin(1)
    advance=torch.minimum(advance,eos).to(torch.int32)
    output=torch.where(columns<advance[:,None],choices,-1)
    return output,advance

class KVTransaction:
    """Reference semantics for a window with isolated speculative staging.

    No production backend is implied. Real native implementation must bind
    equivalent device buffers and obey the same commit/abort ownership rule.
    """
    def __init__(self,window,prefix):
        if window<1:raise ValueError('window')
        self.window=window;self.length=len(prefix);self.committed={i:v for i,v in enumerate(prefix)if i>=self.length-window};self.pending=None
    def begin(self,values):
        if self.pending is not None:raise RuntimeError('transaction already open')
        self.pending=list(values)
    def view(self,query):
        if self.pending is None or not 0<=query<len(self.pending):raise ValueError('query')
        end=self.length+query
        return [self.committed[i]if i<self.length else self.pending[i-self.length]for i in range(max(0,end-self.window+1),end+1)]
    def commit(self,advance):
        if self.pending is None or not 0<=advance<=len(self.pending):raise ValueError('advance')
        self.committed.update({self.length+i:v for i,v in enumerate(self.pending[:advance])});self.length+=advance
        self.committed={i:v for i,v in self.committed.items()if i>=self.length-self.window};self.pending=None
    def abort(self):
        if self.pending is None:raise RuntimeError('no transaction')
        self.pending=None

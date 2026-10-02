"""Same-input, all-rank attention/MoE boundaries with plain-logit bit gates."""
import hashlib
import json
from pathlib import Path


def run(llm, prompt, continuation, document, layers):
    import torch
    from vllm import SamplingParams
    positions=(0,16,48) if layers==70 else (0,)
    arms={}
    for arm,method in [('reference','production_policy_reference'),('canonical','production_policy')]:
        for position in positions:
            forced=prompt['prompt_token_ids']+continuation[:position]
            forced_next=continuation[position]
            captures={}
            for mode in ('plain','boundaries'):
                llm.collective_rpc('native_configure',args=(False,))
                llm.collective_rpc(method,args=(document,))
                tag=f'boundary-{arm}-{mode}-teacher-{position}'
                records=llm.collective_rpc('production_quality_capture',args=(tag,mode))
                output=llm.generate([{'prompt_token_ids':forced}],SamplingParams(
                    temperature=0,max_tokens=2,ignore_eos=True,allowed_token_ids=[forced_next]),use_tqdm=False)[0].outputs[0]
                llm.collective_rpc('production_quality_capture',args=(None,))
                path=Path(next(r['path'] for r in records if r['rank']==0))
                data=torch.load(path,weights_only=True,map_location='cpu')
                captures[mode]=(data,list(output.token_ids),records)
            plain,stamped=captures['plain'][0],captures['boundaries'][0]
            bitwise=all(torch.equal(plain[k].contiguous().view(torch.uint8),stamped[k].contiguous().view(torch.uint8))
                        for k in ('logits','input_ids','positions'))
            bitwise &= captures['plain'][1]==captures['boundaries'][1]
            if not bitwise:raise AssertionError('Boundary sampling changed complete logits: '+arm)
            hashes={}
            expected={(layer,stage) for layer in range(layers) for stage in ('qkv','attention','layer_output')}
            expected|={(layer,'moe_local') for layer in range(1,layers)}
            for record in captures['boundaries'][2]:
                rank=record['rank'];base=Path(record['path'])
                path=base.with_name(base.stem+f'-boundaries-rank{rank}.json')
                data=json.loads(path.read_text())
                if data['rank']!=rank or data['layers']!=layers or data['module_hooks_used']:
                    raise AssertionError('Wrong boundary producer schema')
                if data['input_ids']!=stamped['input_ids'].flatten().tolist() or data['positions']!=stamped['positions'].flatten().tolist():
                    raise AssertionError('Boundary input differs from plain teacher input')
                rows=data['records'];keys={(r['layer'],r['stage']) for r in rows}
                if len(rows)!=len(expected) or keys!=expected:raise AssertionError('Incomplete/duplicate boundary coverage')
                hashes[rank]={key:(r['shape'],r['dtype'],r['sha256']) for r in rows for key in [(r['layer'],r['stage'])]}
            if set(hashes)!=set(range(8)):raise AssertionError('Missing rank boundary evidence')
            arms[arm,position]=(hashes,stamped)
    compared=0
    for position in positions:
        reference,canonical=arms['reference',position],arms['canonical',position]
        if reference[0]!=canonical[0]:raise AssertionError('Attention/MoE boundary bytes changed')
        for key in ('logits','input_ids','positions'):
            if not torch.equal(reference[1][key].contiguous().view(torch.uint8),canonical[1][key].contiguous().view(torch.uint8)):
                raise AssertionError('Reference/canonical plain output changed')
        compared+=sum(len(rows) for rows in reference[0].values())
    return {'pass':True,'positions':list(positions),'layers':layers,'ranks':8,
            'records_compared':compared,'plain_logits_bitwise_equal':True,'module_hooks_used':False,
            'scope':'Untimed same-input sealed-installer vs canonical source; actual QKV, attention, local MoE, layer output'}

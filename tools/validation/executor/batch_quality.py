"""All-request teacher-forced logits, aligned by explicit request/query ownership."""
import json


def generate_with_ids(llm,prompts,params):
    """Observe vLLM's declared internal/external mapping at admission, not parsing IDs."""
    processor=llm.llm_engine.input_processor
    original=processor.assign_request_id
    mapping={}
    def assign(request):
        original(request)
        mapping[request.request_id]=request.external_req_id
    processor.assign_request_id=assign
    try:
        outputs=llm.generate(prompts,params,use_tqdm=False)
        return outputs,mapping
    finally:
        processor.assign_request_id=original


def attach_owners(frames, outputs, mapping=None):
    request_indices={output.request_id:i for i,output in enumerate(outputs)}
    for frame in frames:
        frame['request_indices']=[None if rid is None else request_indices[
            mapping[rid] if mapping is not None else rid] for rid in frame['request_ids']]
    return frames


def index_rows(frames):
    """A request may advance in B1, B2 or B3; batch packing is not its identity."""
    result={}
    for frame in frames:
        for row,(request,prompt,query) in enumerate(zip(frame['request_indices'],
                frame['request_keys'],frame['logits_indices'])):
            if request is None:
                continue
            if not prompt or not 0 <= query < len(frame['input_ids']):
                raise ValueError('Missing teacher query provenance')
            key=(request,prompt,frame['input_ids'][query],frame['positions'][query])
            if key in result:
                raise ValueError('Duplicate teacher query: '+str(key))
            result[key]=(frame,row)
    return result


def compare_rows(first, second):
    import torch
    from gaudi_kernels.serving.executor.production_quality import compare_logits
    a,b=index_rows(first),index_rows(second)
    checks=[];cache={}
    def load(frame,row):
        path=frame['path']
        if path not in cache:
            cache[path]=torch.load(path,weights_only=True,map_location='cpu')['logits']
        return cache[path][row:row+1]
    for key in sorted(set(a)&set(b)):
        af,ar=a[key];bf,br=b[key];x,y=load(af,ar),load(bf,br)
        check=compare_logits(x,y,contract='fp32_arithmetic_v1')
        check['bitwise_equal']=torch.equal(x.view(torch.uint8),y.view(torch.uint8))
        checks.append({'request':key[0],'prompt_sha256':key[1],'input_id':key[2],'position':key[3],
            'reference_path':af['path'],'candidate_path':bf['path'],'check':check})
    return checks


def coverage(checks,batch):
    return all(sum(row['request']==i for row in checks)>=4 for i in range(batch))


def compare(llm, batch, tokens, control, candidate, root):
    from vllm import SamplingParams
    from gaudi_kernels.engine.selection import PolicySelection
    control=PolicySelection.from_dict(control).to_dict()
    candidate=PolicySelection.from_dict(candidate).to_dict()
    if control['engine']['runtime'] != candidate['engine']['runtime']:
        raise ValueError('Teacher control must retain the same runtime and runner')
    llm.collective_rpc('native_configure',args=(False,False,False,'compact'))
    llm.collective_rpc('production_policy',args=(control,))
    prompts=[{'prompt_token_ids':(tokens*128)[:124]+[tokens[i%len(tokens)]]} for i in range(batch)]
    anchors=[o.outputs[0].token_ids[0] for o in llm.generate(prompts,
        SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)]
    (root/f'b{batch}-teacher-plan.json').write_text(json.dumps({
        'prompts':prompts,'anchors':anchors,'control':control,'candidate':candidate},indent=2)+'\n')
    params=[SamplingParams(temperature=0,max_tokens=8,ignore_eos=True,allowed_token_ids=[v]) for v in anchors]
    records={}
    for name,document in (('control',control),('candidate',candidate)):
        llm.collective_rpc('production_policy',args=(document,))
        llm.collective_rpc('production_quality_capture',args=(f'b{batch}-{name}-plain','batch',batch))
        outputs,mapping=generate_with_ids(llm,prompts,params)
        plain=llm.collective_rpc('production_quality_capture',args=(None,))
        plain_frames=attach_owners(next(r['captured_forwards'] for r in plain if r['rank']==0),outputs,mapping)
        llm.collective_rpc('production_quality_capture',args=(f'b{batch}-{name}','batch_boundaries',batch))
        outputs,mapping=generate_with_ids(llm,prompts,params)
        ranks=llm.collective_rpc('production_quality_capture',args=(None,))
        frames=attach_owners(next(r['captured_forwards'] for r in ranks if r['rank']==0),outputs,mapping)
        boundaries=[]
        for rank in range(8):
            path=root/'quality-logits'/f'b{batch}-{name}-boundaries-rank{rank}.json'
            doc=json.loads(path.read_text())
            if len(doc['input_ids'])<batch or len(doc['records'])!=4*doc['layers']-1:
                raise ValueError('Incomplete all-request producer coverage')
            boundaries.append({'rank':rank,'path':str(path),'records':len(doc['records']),
                               'input_rows':len(doc['input_ids'])})
        records[name]={'frames':plain_frames,'boundary_frames':frames,
                       'plain_vs_boundaries':[],'boundaries':boundaries}
        (root/f'b{batch}-{name}-diagnostics.json').write_text(json.dumps(records[name],indent=2)+'\n')
        diagnostics=compare_rows(plain_frames,frames)
        records[name]['plain_vs_boundaries']=diagnostics
        (root/f'b{batch}-{name}-diagnostics.json').write_text(json.dumps(records[name],indent=2)+'\n')
        if not coverage(diagnostics,batch):
            raise ValueError('Insufficient request/query alignment for producer diagnostics')
        if not all(row['check']['pass'] for row in diagnostics):
            raise ValueError('Producer diagnostics changed teacher logits beyond the FP32 quality contract')
    checks=compare_rows(records['control']['frames'],records['candidate']['frames'])
    passed=coverage(checks,batch) and all(row['check']['pass'] for row in checks)
    result={'pass':passed,'batch':batch,'anchors':anchors,'checks':checks,'records':records,
        'scope':'all request/query ownership in matched teacher-forced forwards; not complete-answer quality or TPS'}
    (root/f'b{batch}-teacher.json').write_text(json.dumps(result,indent=2)+'\n')
    if not passed:
        raise RuntimeError('All-request teacher-forced logits gate failed')
    return result

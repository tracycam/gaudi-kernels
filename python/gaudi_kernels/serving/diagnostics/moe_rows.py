"""Probe compact MoE rows beyond serving admission using unchanged loaded ELFs.

Direct composition leaves production dispatch guards untouched. Compares the
complete local consumer chain with production, row permutation and zero routes.
Independent FP32 gate/up samples are reported, without a precision-choice gate.
"""
import json
import time


def on_worker(worker, plan):
    import torch
    import habana_frameworks.torch.core as hc
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.serving.models.mimo_mxfp4 import NativeExpertTP
    from gaudi_kernels.serving.executor.batch_ops import moe
    from gaudi_kernels.serving.executor.gp_scale_tail_runtime import operator
    from gaudi_kernels.serving.executor.precision_ops import combine
    from gaudi_kernels.serving.executor.packing import unpack, unpack_acc32_lanes
    from gaudi_kernels.serving.diagnostics.fp32_reference import dot_error_bound

    rows = tuple(plan['rows'])
    if (worker.model_runner.input_batch.num_reqs or context().native.active or
            not rows or any(type(n) is not int or n not in (8, 12, 16, 24) for n in rows)):
        raise ValueError('Idle worker, inactive native serving and bounded rows required')
    experts = [m for m in worker.model_runner.model.modules() if isinstance(m, NativeExpertTP)]
    if not experts:
        raise ValueError('No actual TP expert weights loaded')
    e = experts[0]
    gp_call = operator()
    root = context().startup.run_dir/'moe-row-coverage'
    root.mkdir(exist_ok=True)
    report = dict(rank=worker.rank, checks=[], timing=[], pass_=False,
        production_guard_changed=False, weight_dtype=str(e.gp.dtype),
        scope='TP-local real-weight complete GP/gate/down/combine probe; no collective, model-quality or service TPS claim')
    def save():
        value = dict(report)
        value['pass'] = value.pop('pass_')
        (root/f'rank{worker.rank}.json').write_text(json.dumps(value, indent=2)+'\n')
    def sync():
        hc.mark_step()
        torch.hpu.synchronize()
    def compact(x, ids, routing):
        # Keep Lazy materialization identical to production. Removing clones
        # is a different experiment, not part of row-coverage qualification.
        x, ids, routing = x.clone(), ids.int().clone(), routing.float().clone()
        p = gp_call(e.gp, e.gs, x, e.table, ids)
        gate = torch.ops.unified_batch.gate(p, ids)
        d = torch.ops.gaudi_down_activation.broadcast(e.dp, e.ds, gate, e.table, ids)
        return combine(d, routing, e.directions, 6144, 8), p.clone()
    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    generator = torch.Generator().manual_seed(103124)
    with torch.inference_mode():
        for n in rows:
            x_cpu = (torch.randn(n, 6144, generator=generator)*.1).bfloat16()
            ids_cpu = (torch.arange(n*8).reshape(n, 8)*37 % 384).int()
            weights_cpu = torch.rand(n, 8, generator=generator)
            weights_cpu /= weights_cpu.sum(-1, keepdim=True)
            inputs = [v.to(worker.model_runner.device) for v in (x_cpu, ids_cpu, weights_cpu)]
            sync()
            graphs, outputs = {}, {}
            for mode in ('broadcast', 'compact'):
                graph = torch.hpu.HPUGraph()
                with torch.hpu.graph(graph):
                    if mode == 'compact':
                        y, partial = compact(*inputs)
                    else:
                        y = moe(*inputs, e.gp, e.gs, e.dp, e.ds, e.table, e.directions, mode='broadcast')
                        partial = None
                    consumed = y + .03125
                sync()
                graphs[mode], outputs[mode] = graph, (y, consumed, partial)
                graph.replay()
                sync()
            reference = None
            for state in ('uniform', 'permuted', 'hot', 'zero'):
                if state == 'uniform':
                    raw = (x_cpu, ids_cpu, weights_cpu)
                elif state == 'permuted':
                    raw = tuple(v.flip(0) for v in (x_cpu, ids_cpu, weights_cpu))
                elif state == 'hot':
                    raw = (-x_cpu, torch.arange(376,384).expand(n,8).int(), weights_cpu)
                else:
                    raw = (x_cpu, ids_cpu, torch.zeros_like(weights_cpu))
                for destination, source in zip(inputs, raw):
                    destination.copy_(source)
                sync()
                cpu = {}
                for mode in ('broadcast', 'compact'):
                    graphs[mode].replay()
                    sync()
                    cpu[mode] = tuple(v.cpu() if v is not None else None for v in outputs[mode])
                finite = all(bool(v.isfinite().all()) for values in cpu.values() for v in values if v is not None)
                equal = all(same(a,b) for a,b in zip(cpu['broadcast'][:2],cpu['compact'][:2]))
                metamorphic = (state != 'permuted' or same(cpu['compact'][0],reference.flip(0))) and (
                    state != 'zero' or bool((cpu['compact'][0] == 0).all()))
                if state == 'uniform':
                    reference = cpu['compact'][0]
                    # Independently decode expert0 columns, with actual scales.
                    # CPU-only temporary expansion; no persistent/VRAM cache.
                    w = e.gp[:6144].cpu().numpy().reshape(1,6144,256)
                    s = e.gs[:192].cpu().numpy().reshape(1,192,512)
                    packed, scales = unpack(w,s)
                    columns = torch.arange(512)
                    packed = torch.from_numpy(packed)[columns]
                    code = torch.empty(512,6144,dtype=torch.long)
                    code[:,0::2],code[:,1::2] = (packed & 15).long(),(packed >> 4).long()
                    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
                    scale = torch.pow(2.,torch.from_numpy(scales)[columns].float()-127).repeat_interleave(32,-1)
                    decoded = lut[code]*scale
                    fp32 = decoded @ x_cpu[0].float()
                    bound = dot_error_bound(decoded*x_cpu[0].float())
                    raw_partial = cpu['compact'][2].reshape(n,8,3,512)[0,0]
                    actual = torch.from_numpy(unpack_acc32_lanes(raw_partial.numpy())).sum(0)[columns]
                    # A separately materialized GP tests whether a graph's
                    # consumer reused a supposedly retained partial buffer.
                    solo = gp_call(e.gp,e.gs,inputs[0].clone(),e.table,inputs[1].int().clone()).clone()
                    sync()
                    solo_partial = solo.cpu().reshape(n,8,3,512)[0,0]
                    solo_cpu = torch.from_numpy(unpack_acc32_lanes(solo_partial.numpy())).sum(0)
                    torch.save(dict(packed=torch.from_numpy(unpack(w,s)[0]),
                        scales=torch.from_numpy(unpack(w,s)[1]),x=x_cpu[0],
                        graph_partial_abi=raw_partial,standalone_partial_abi=solo_partial,
                        graph_actual=actual,standalone_actual=solo_cpu,reference=fp32),
                        root/f'rank{worker.rank}-r{n}-gp-reference.pt')
                    report.setdefault('fp32_samples',[]).append(dict(rows=n,
                        reference=fp32.tolist(), actual=actual.tolist(),
                        relative_l2=float((actual-fp32).norm()/fp32.norm().clamp_min(1e-30)),
                        standalone_relative_l2=float((solo_cpu-fp32).norm()/fp32.norm().clamp_min(1e-30)),
                        graph_standalone_bits_equal=same(actual,solo_cpu),
                        fp32_forward_bound_pass=bool(((actual-fp32).abs()<=bound).all() and
                                                     ((solo_cpu-fp32).abs()<=bound).all()),
                        scope='Independent FP32 dot products; legal accumulation-tree differences reported, precision threshold deferred'))
                    save()
                    if not report['fp32_samples'][-1]['fp32_forward_bound_pass']:
                        raise ValueError('GP error exceeds both FP32 accumulation trees: inspect indexing/layout/scales')
                check = dict(rows=n,state=state,finite=finite,production_bits_equal=equal,metamorphic_pass=metamorphic)
                report['checks'].append(check)
                torch.save(dict(inputs=raw,outputs=cpu),root/f'rank{worker.rank}-r{n}-{state}.pt')
                save()
                if not (finite and equal and metamorphic):
                    raise ValueError('Unchanged-ELF row mapping/consumer regression: '+str(check))
                if state in ('uniform','hot'):
                    for trial in range(2):
                        for mode in ('broadcast','compact','compact','broadcast'):
                            began = time.perf_counter_ns()
                            for _ in range(8):
                                graphs[mode].replay(asynchronous=True)
                            sync()
                            report['timing'].append(dict(rows=n,state=state,trial=trial,variant=mode,
                                replay_wall_us=(time.perf_counter_ns()-began)/8000))
                    save()
            del graphs, outputs, inputs
            sync()
    report['pass_'] = True
    save()
    return dict(report, **{'pass': True})

"""Versioned acceptance roles. Does not rewrite or reclassify historical runs."""

VERSION = 'fp32_arithmetic_v1'


def validate_ocp_requirement(contract, model_reference, required):
    if required and (contract != VERSION or not model_reference):
        raise ValueError('required OCP quality needs fp32_arithmetic_v1 and --fp32-ocp-model-reference')


def plan(policies, reference_policy, candidate_policies, *, require_fp32_ocp_quality=False):
    candidates = candidate_policies.split(',') if candidate_policies else [policies[-1]]
    if (not reference_policy or reference_policy not in policies
            or reference_policy.split('+')[0] != 'bf16_fp32'):
        raise ValueError('fp32_arithmetic_v1 requires an explicit resident A16 quality reference policy')
    if (not candidates or len(set(candidates)) != len(candidates)
            or any(p not in policies or not p.startswith('decode_a8_') for p in candidates)):
        raise ValueError('v1 candidates must be distinct resident block-A8 device policies')
    if 'norm_fp32' not in reference_policy.split('+')[1:] or any('norm_fp32' not in p.split('+')[1:] for p in candidates):
        raise ValueError('v1 reference and candidates require +norm_fp32; vendor norm is a performance control only')
    return dict(contract=VERSION, reference_policy=reference_policy, candidate_policies=candidates,
                require_fp32_ocp_quality=bool(require_fp32_ocp_quality),
                performance_control_policies=[p for p in policies if p not in [reference_policy, *candidates]],
                teacher_policies=[reference_policy, *candidates],
                scope='explicit new-run contract; original FP64 records remain diagnostic and unchanged')


def role(policy, contract_plan):
    if policy in contract_plan['candidate_policies']:
        return 'candidate'
    if policy == contract_plan['reference_policy']:
        return 'precision_reference'
    if policy.startswith('cpu_'):
        return 'fp32_reference' if policy.split('+')[0].endswith('_fp32_v1') else 'fp64_diagnostic'
    return 'performance_control'


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _finite_check(row):
    check = row.get('check', {})
    return check.get('pass') is True and check.get('finite') is True


def _requested_reduction(policy):
    suffixes = set(policy.split('+')[1:])
    return 'neumaier_isa_fp32' if 'qkv_neumaier_isa' in suffixes else 'neumaier_fp32' if 'qkv_neumaier' in suffixes else 'sequential'


def staged_gate(row, layers):
    """All selected layers/ranks at the exact teacher input; no sampled-row pass."""
    failures = []
    workers = row.get('same_input_qkv_audit', [])
    if len(workers) != 8 or {x.get('rank') for x in workers} != set(range(8)):
        return dict(passed=False, reasons=['eight distinct ranks required'], checked_layers=0)
    relation = row.get('plain_vs_audit_logits', {})
    if relation.get('all_bits_equal') is not True or relation.get('same_decode_input') is not True:
        failures.append('plain model and staged-audit model relation is incomplete')
    count = 0
    for worker in workers:
        records = worker.get('same_input_qkv_audit', [])
        if 'norm_qkv_grid24' in row['policy'].split('+')[1:]:
            grid=[r for r in records if 'norm_grid24_producer' in r]
            expected=worker.get('expected_grid24_qkv_layers',[])
            if (not expected or len(set(expected))!=len(expected) or len(grid)!=len(expected)
                    or {r.get('layer')for r in grid}!=set(expected)):
                failures.append(f'rank {worker["rank"]}: incomplete actual grid24 producer layer coverage')
            for record in grid:
                producer=record['norm_grid24_producer']
                if (producer.get('passed') is not True
                        or producer.get('implementation',{}).get('guid')!='gk_norm_block128_grid24_v1'
                        or any(producer.get(k,{}).get('all_bits_equal') is not True for k in
                            ('plain_vs_staged_residual','old_vs_grid_residual','old_vs_grid_q','old_vs_grid_scale'))):
                    failures.append(f'rank {worker["rank"]}: incomplete grid24 producer relation')
        selected = worker.get('selected_qkv_layers', [])
        if (len(selected) != layers or len(set(selected)) != layers or len(records) != layers
                or {r.get('layer') for r in records} != set(selected)):
            failures.append(f'rank {worker["rank"]}: incomplete/duplicate selected-layer coverage')
        for record in records:
            report = record.get('fp32_contract', {})
            frame = record.get('frame', {})
            valid = (record.get('quality_contract') == VERSION and record.get('tag') == row.get('audit_tag')
                     and record.get('policy') == row['policy'].split('+')[0] and record.get('same_input') is True
                     and frame.get('input_ids') == [row['forced_decode_token']]
                     and frame.get('positions') == [row['decode_position']]
                     and record.get('reduction_policy') == _requested_reduction(row['policy'])
                     and report.get('reduction') == _requested_reduction(row['policy'])
                     and report.get('contract') == 'block_fp8_fp32_v1' and report.get('passed') is True
                     and report.get('numerical_passed') is True and report.get('implementation_qualified') is True
                     and record.get('source_layout') is not None
                     and record.get('plain_vs_staged', {}).get('all_bits_equal') is True
                     and _sha256(record.get('input_sha256'))
                     and _sha256(record.get('checkpoint_weight_sha256'))
                     and _sha256(record.get('checkpoint_scales_sha256'))
                     and record.get('checkpoint_weight_sha256') == report.get('checkpoint_weight_sha256')
                     and record.get('checkpoint_scales_sha256') == report.get('checkpoint_scales_sha256'))
            if not valid:
                failures.append(f'rank {worker["rank"]}: {record.get("layer")} lacks bound staged/implementation evidence')
            count += 1
    return dict(passed=not failures, reasons=failures, checked_layers=count,
                expected_layers=8 * layers, scope='all QKV layers, all ranks, exact teacher frame; independent staged reference')


def classify(result, contract_plan, positions, layers):
    """Keep FP64/reference-vs-reference rows out of candidate acceptance."""
    if result.get('quality_contract') != VERSION:
        raise ValueError('cannot apply v1 classification to a legacy result')
    candidates = contract_plan['candidate_policies']
    rows = result['teacher_forced']['rows']
    required = {(policy, position) for policy in candidates for position in positions}
    selected = [r for r in rows if r.get('quality_role') == 'candidate']
    coverage = {(r['policy'], r['position']) for r in selected}
    complete = len(selected) == len(required) and coverage == required
    teacher_pass = complete and all(r.get('same_decode_input') is True and _finite_check(r) for r in selected)
    audits = [dict(policy=r['policy'], position=r['position'], **staged_gate(r, layers)) for r in selected]
    staged_pass = complete and all(r['passed'] for r in audits)
    references = [r for r in result.get('same_contract_reference', []) if r.get('required_for_acceptance')]
    require_ocp = contract_plan.get('require_fp32_ocp_quality', False)
    expected_bases = {'cpu_native_a8_fp32_v1'} | ({'cpu_ocp_a8_fp32_v1'} if require_ocp else set())
    unknown_required = any(r.get('reference_policy', '').split('+')[0] not in expected_bases for r in references)
    def reference_pass(base):
        matched = [r for r in references if r.get('reference_policy', '').split('+')[0] == base]
        complete = len(matched) == len(required) and {(r['policy'], r['position']) for r in matched} == required
        return complete and all(r.get('same_decode_input') is True and _finite_check(r) for r in matched)
    matched_pass = reference_pass('cpu_native_a8_fp32_v1') and not unknown_required
    ocp_pass = reference_pass('cpu_ocp_a8_fp32_v1') if require_ocp else None
    tasks = [r for r in result.get('chat_quality', []) if r['policy'] in candidates]
    tasks_complete = len(tasks) == 3 * len(candidates) and {
        (r['policy'], r['task']) for r in tasks} == {(p, t) for p in candidates for t in ('arithmetic', 'retrieval', 'chinese')}
    tasks_pass = tasks_complete and all(r['pass'] and r.get('normal_eos_pass') for r in tasks)
    return dict(contract=VERSION, passed=bool(teacher_pass and staged_pass and matched_pass and tasks_pass and (not require_ocp or ocp_pass)),
                candidate_teacher_pass=bool(teacher_pass), staged_qkv_pass=bool(staged_pass),
                matched_fp32_reference_pass=bool(matched_pass), normal_eos_and_tasks_pass=bool(tasks_pass),
                require_fp32_ocp_quality=require_ocp, original_ocp_fp32_reference_pass=ocp_pass,
                recipe_relation_status='COMPLETE' if staged_pass else 'INCOMPLETE_OR_ARITHMETIC_FAILURE',
                audits=audits, ignored_reference_or_control_rows=len(rows) - len(selected),
                thresholds=dict(max_kl=.01, raw_logit_relative_l2='diagnostic'),
                scope='finite/KL, full staged QKV evidence, normal EOS/tasks; optional required OCP M1 reference with shared prefill/KV; no old-record PASS backfill')

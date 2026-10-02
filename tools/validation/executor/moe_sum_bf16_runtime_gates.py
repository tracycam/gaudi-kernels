def capture_gate(policy, states):
    requested = 'moe_sum_bf16' in policy.split('+')[1:]
    expected = 'moe_final_bf16' if requested else 'baseline'
    passed = len(states) == 8 and all((state.get('moe_sum_bf16', {}).get('policy') == expected for state in states))
    if requested:
        passed = passed and all((state['moe_sum_bf16'].get('operator') == 'precision_fix::sum_fb' and state['moe_sum_bf16'].get('python_capture_counts', {}).get('sum_fb_capture_calls', 0) > 0 and (state['moe_sum_bf16'].get('python_capture_counts', {}).get('eligible_without_matching_ag', 0) == 0) and (state['moe_sum_bf16'].get('python_capture_counts', {}).get('scope_identity_or_rank_mismatch', 0) == 0) for state in states))
    return {'pass': bool(passed), 'requested': requested, 'expected_policy': expected, 'scope': 'all-rank Python capture binding; compiled sum_fb presence/cast removal and numerical gate still required'}

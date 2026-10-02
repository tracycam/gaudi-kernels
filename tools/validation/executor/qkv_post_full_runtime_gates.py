def attention_scope(suffixes):
    suffixes = set(suffixes)
    if 'qkv_post_full' in suffixes and 'qkv_post' not in suffixes:
        raise ValueError('qkv_post_full requires explicit qkv_post')
    return 'swa128_or_full' if 'qkv_post_full' in suffixes else 'swa128'

def capture_gate(policy, states):
    requested = 'qkv_post_full' in policy.split('+')[1:]
    if not requested:
        return {'pass': True, 'requested': False, 'scope': 'existing postprocess gate remains unchanged'}
    records = []
    for state in states:
        post = state.get('qkv_postprocess', {})
        counts = post.get('by_policy', {}).get('fused', {})
        expected = (post.get('prepared_attention_counts') or {}).get('full', 0)
        actual = post.get('eligible_layers_by_policy', {}).get('fused', {}).get('full', {})
        passed = post.get('policy') == 'fused' and post.get('attention_scope') == 'swa128_or_full' and (expected > 0) and (len(actual) == expected) and all((v > 0 for v in actual.values())) and (counts.get('eligible_full_calls', 0) > 0)
        records.append({'rank': state.get('rank'), 'pass': bool(passed), 'expected_full_layers': expected, 'captured_full_layers': len(actual)})
    return {'pass': len(states) == 8 and all((row['pass'] for row in records)), 'requested': True, 'ranks': records, 'scope': 'all statically admitted full-attention modules reached Python capture on each rank; not executed kernel counts or numerical qualification'}

def arithmetic_baseline(policy, suffix):
    """Remove one complete suffix token; never strip qkv_post from *_full."""
    (base, *parts) = policy.split('+')
    key = suffix.removeprefix('+')
    if key not in parts:
        return None
    if key == 'qkv_post' and 'qkv_post_full' in parts:
        return None
    result = []
    for value in parts:
        if value != key:
            result.append(value)
        elif key == 'gp_fold':
            result.append('gp_vec')
    return '+'.join([base, *result])

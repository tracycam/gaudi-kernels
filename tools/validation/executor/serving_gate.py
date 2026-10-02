"""Performance labels require the same execution and quality gates as acceptance."""
def performance_qualified(layers,run):
    return (layers==70 and all(run.get(k) is True for k in (
        'token_count_pass','bitwise_token_match','quality_pass','all_capture_gates_pass'))
        and (not run['active'] or run.get('native_used_all_ranks') is True))

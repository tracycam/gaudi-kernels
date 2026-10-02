#!/usr/bin/env python3
"""Join executed device trace GUIDs to registered library GUIDs exactly."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(registry, trace):
    registered = {}
    for library in registry['libraries']:
        for guid in library['registered_guids']:
            if guid in registered:
                raise ValueError('Ambiguous GUID registry')
            registered[guid] = library['sha256']
    counts = Counter()
    nodes = {}
    with trace.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get('ph') != 'B':
                continue
            operation = event.get('args', {}).get('op')
            if operation in registered:
                counts[operation] += 1
                nodes.setdefault(operation, set()).add((event['args'].get('recipe_id'), event.get('name')))
    return {
        'trace': str(trace), 'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
        'executed_external_guids': [{'guid': guid, 'library_sha256': registered[guid],
            'physical_begin_records': counts[guid], 'distinct_recipe_node_labels': len(nodes[guid])}
            for guid in sorted(counts)],
        'registered_but_not_observed': sorted(set(registered) - set(counts)),
        'scope': 'Exact GUID match in device begin records; not host selection inference. '
                 'Counts include multiple cores and replay steps. Unobserved here does not mean dead: '
                 'this trace is decode-only and cannot retire prefill/reference/fallback kernels.',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(json.loads(a.registry.read_text()), a.trace)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'executed_external_guids': len(result['executed_external_guids']),
                      'unobserved_registered': len(result['registered_but_not_observed'])}))

#!/usr/bin/env python3
"""Read-only per-file disposition of the locally modified hardware plugin.

This is a review queue, not a claim that patches were upstreamed or removed.
Every decision is attached to real changed paths and a SHA of the full patch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(repository, *arguments):
    return subprocess.check_output(['git', '-C', str(repository), *arguments], text=True)


def classify(path):
    if path.startswith(('bench/', 'benchmarks/', 'tools/', 'oracle/', 'docs/', 'tests/', 'experiments/')):
        return 'archive', 'Diagnostic/reference/benchmark asset; retain evidence, remove from production dependency'
    if path.startswith('vllm_gaudi/models/'):
        return 'plugin', 'Model registration/loader adaptation belongs in the model-specific engine plugin'
    if path.startswith(('vllm_gaudi/attention/', 'vllm_gaudi/v1/attention/')):
        return 'upstream', 'Generic attention correctness or device compatibility; preserve until accepted upstream'
    if path.startswith(('vllm_gaudi/ops/', 'vllm_gaudi/extension/')):
        return 'plugin', 'Device precision/quantization policy must be replaced by explicit engine ops; generic fixes split upstream'
    if path.startswith('vllm_gaudi/v1/worker/'):
        return 'upstream', 'Generic runner/input/metadata behavior; requires review and a worker/runner extension boundary'
    if path in ('vllm_gaudi/__init__.py', 'vllm_gaudi/platform.py'):
        return 'plugin', 'Register model/worker through public plugin and worker_cls; no import-time rewrite'
    return 'review', 'Mixed build/package or cross-cutting change requires manual separation'


def audit(repository, base):
    rows = []
    for revision in git(repository, 'rev-list', '--reverse', base + '..HEAD').splitlines():
        subject = git(repository, 'show', '-s', '--format=%s', revision).strip()
        names = git(repository, 'diff-tree', '--no-commit-id', '--name-only', '-r', revision).splitlines()
        patch = subprocess.check_output(['git', '-C', str(repository), 'show', '--format=fuller', revision])
        paths = []
        for name in names:
            disposition, rationale = classify(name)
            paths.append({'path': name, 'disposition': disposition, 'rationale': rationale})
        rows.append({'commit': revision, 'subject': subject,
                     'patch_sha256': hashlib.sha256(patch).hexdigest(), 'paths': paths})
    return {'repository': str(repository.resolve()), 'base': git(repository, 'rev-parse', base).strip(),
        'head': git(repository, 'rev-parse', 'HEAD').strip(), 'commit_count': len(rows), 'commits': rows,
        'scope': 'Per-file migration review queue; no upstream submission, deletion or plugin qualification yet'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--base', default='origin/main')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.repository, a.base)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'commit_count': result['commit_count'], 'output': str(a.output)}))

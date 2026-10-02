"""Copy a completed owned model case locally and verify remote/frozen hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runner-input', type=Path, action='append', default=[])
    a = p.parse_args()
    if not a.case or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.case):
        p.error('case must be a single safe experiment name')
    ssh = ['ssh', '-S', a.control, '-p', str(a.port)]
    script = '''from pathlib import Path
import json,hashlib
r=Path(REMOTE);c=CASE
run=json.loads((r/(c+'.exit.json')).read_text())
assert not run.get('remaining_owned_processes'), 'owned workers still running'
assert run.get('cleanup',{}).get('reaped'), 'owned launcher not reaped'
paths=[p for p in (r/c).rglob('*') if p.is_file()]+[p for p in r.glob(c+'.*') if p.is_file()]
print(json.dumps({str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}))
'''.replace('REMOTE', repr(a.remote_root)).replace('CASE', repr(a.case))
    manifest = json.loads(subprocess.check_output(ssh + [a.host, 'python3 -'],
                                                  input=script.encode(), timeout=180))
    case, runner = a.out / a.case, a.out / (a.case + '-runner')
    case.mkdir(parents=True, exist_ok=True)
    runner.mkdir(parents=True, exist_ok=True)

    def copy(source, target):
        subprocess.run(['rsync', '-a', '--checksum', '--protect-args', '-e', shlex.join(ssh),
                        a.host + ':' + a.remote_root + '/' + source, str(target) + '/'],
                       check=True, timeout=300)

    copy(a.case + '/', case)
    for name in manifest:
        if '/' not in name:
            copy(name, runner)
    for name, record in manifest.items():
        local = case / name.split('/', 1)[1] if '/' in name else runner / name
        if local.stat().st_size != record['bytes'] or hashlib.sha256(local.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('remote/local asset mismatch: ' + name)
    frozen = json.loads((case / 'source/sha256.json').read_text())
    for name, digest in frozen.items():
        if hashlib.sha256((case / 'source' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen source/library mismatch: ' + name)
    for source in a.runner_input:
        (runner / source.name).write_bytes(source.read_bytes())
    destination = runner / 'remote-sha256.json'
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    summary = dict(case=a.case, files=len(manifest), bytes=sum(v['bytes'] for v in manifest.values()),
                   frozen_sources=len(frozen), remote_and_frozen_hashes_verified=True,
                   manifest_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), local=str(case))
    (runner / 'seal-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

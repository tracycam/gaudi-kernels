"""Verify remote byte identity, frozen source identity, and a local canonical copy."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    remote=json.loads((a.raw/'remote-sha256.json').read_text())
    for name,entry in remote['files'].items():
        local=a.raw/'remote'/name
        assert local.stat().st_size==entry['bytes'],name
        assert sha(local)==entry['sha256'],name
    identity=json.loads((a.raw/'remote/source-identity.json').read_text())
    for name,want in identity['files_sha256'].items():
        assert sha(a.raw/'remote'/name)==want,name
    records={}
    for source in sorted(a.raw.rglob('*')):
        if not source.is_file():continue
        name=str(source.relative_to(a.raw))
        target=a.canonical/name
        digest=sha(source)
        assert target.stat().st_size==source.stat().st_size,name
        assert sha(target)==digest,name
        records[name]=dict(bytes=source.stat().st_size,sha256=digest)
    report=dict(status='PASS_REMOTE_SOURCE_CANONICAL_SHA256',
                canonical_path=str(a.canonical),remote_root=remote['root'],
                device_source_commit=identity['git_commit'],
                verified_remote_files=len(remote['files']),
                verified_frozen_sources=len(identity['files_sha256']),
                file_count=len(records),total_bytes=sum(v['bytes']for v in records.values()),
                files=records)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='files'}))


if __name__=='__main__':main()

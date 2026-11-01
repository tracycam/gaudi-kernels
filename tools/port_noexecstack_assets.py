"""Copy retained operator ELFs with a non-executable host stack.

Binary TPC payloads embedded by objcopy are data, not host stack trampolines.
Old linkers infer an executable stack when these input objects lack GNU-stack
notes. New builds must use -Wl,-z,noexecstack. This explicit migration preserves
original artifacts and records new hashes; it never alters loaded ISA/data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def nonexecuting_stack(raw):
    if raw[:6] != b'\x7fELF\x02\x01' or len(raw) < 64:
        raise ValueError('Expected ELF64 little-endian operator')
    offset, = struct.unpack_from('<Q', raw, 32)
    size, count = struct.unpack_from('<HH', raw, 54)
    if size != 56 or offset + size * count > len(raw):
        raise ValueError('Invalid program header table')
    patches = []
    result = bytearray(raw)
    for index in range(count):
        at = offset + index * size
        kind, flags = struct.unpack_from('<II', raw, at)
        if kind != 0x6474e551 or not flags & 1:
            continue
        filesz, memsz = struct.unpack_from('<QQ', raw, at + 32)
        if filesz or memsz:
            raise ValueError('GNU_STACK unexpectedly contains a payload')
        struct.pack_into('<I', result, at + 4, flags & ~1)
        patches.append(dict(offset=at+4, before=flags, after=flags & ~1))
    # Every changed byte must belong to a reported program-header flag.
    permitted = {p['offset'] + byte for p in patches for byte in range(4)}
    if any(a != b and i not in permitted for i, (a, b) in enumerate(zip(raw, result))):
        raise AssertionError('Unexpected executable/data payload mutation')
    return bytes(result), patches


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--output-manifest', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_manifest.exists():
        raise ValueError('Output manifest already exists')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    document = json.loads(args.manifest.read_text())
    report = {'scope':'Explicit host ELF stack metadata migration; not arithmetic or ISA optimization',
              'source_manifest_sha256':sha(args.manifest.read_bytes()), 'artifacts':{}}
    for key, entry in document['artifacts'].items():
        original = Path(entry['path'])
        raw = original.read_bytes()
        if sha(raw) != entry['sha256']:
            raise ValueError('Retained artifact checksum mismatch: '+key)
        if key == 'vendor.tpc':
            continue  # Do not rewrite official SDK assets.
        if raw.startswith(b'\x7fELF'):
            updated, patches = nonexecuting_stack(raw)
        elif key.endswith(('.tpc', '.torch')):
            raise ValueError('Operator is not an ELF: '+key)
        else:
            updated, patches = raw, []  # Retained host build metadata.
        target = args.output_dir / original.name
        target.write_bytes(updated)
        entry.update(path=str(target.resolve()), sha256=sha(updated))
        report['artifacts'][key] = dict(original=str(original), original_sha256=sha(raw),
                                      path=entry['path'], sha256=entry['sha256'], patches=patches,
                                      all_other_bytes_identical=True)
    document['startup']['run_dir'] = str(args.run_dir.resolve())
    args.output_manifest.write_text(json.dumps(document, indent=2)+'\n')
    report['output_manifest_sha256'] = sha(args.output_manifest.read_bytes())
    (args.output_dir/'migration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'MIGRATED', 'artifacts':len(report['artifacts']),
                      'changed':sum(bool(v['patches']) for v in report['artifacts'].values())}))


if __name__ == '__main__':
    main()

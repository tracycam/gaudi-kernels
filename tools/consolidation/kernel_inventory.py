#!/usr/bin/env python3
"""Inventory registered GUIDs and exact embedded TPC ELF/.text bytes.

This calls only host-library metadata APIs. It does not acquire a device or
infer executed GUIDs. The GuidInfo ctypes layout is checked against the pinned
SDK header (Gaudi2 device ID 3; char[64], uint64 hash, uint32 properties).
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess


class GuidInfo(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char * 64), ('name_hash', ctypes.c_uint64),
                ('properties', ctypes.c_uint32)]


assert ctypes.sizeof(GuidInfo) == 80


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sections(data):
    if len(data) < 52 or data[:4] != b'\x7fELF' or data[5] != 1:
        raise ValueError('Only little-endian ELF32/64 is admitted')
    if data[4] == 2 and len(data) >= 64:
        offset = struct.unpack_from('<Q', data, 40)[0]
        size, count, names_index = struct.unpack_from('<HHH', data, 58)
        entry_format, expected_size = '<IIQQQQIIQQ', 64
    elif data[4] == 1:
        offset = struct.unpack_from('<I', data, 32)[0]
        size, count, names_index = struct.unpack_from('<HHH', data, 46)
        entry_format, expected_size = '<IIIIIIIIII', 40
    else:
        raise ValueError('Invalid ELF class/header')
    if size != expected_size or not count or names_index >= count or offset + count * size > len(data):
        raise ValueError('Invalid ELF section table')
    entries = [struct.unpack_from(entry_format, data, offset + i * size) for i in range(count)]
    names_entry = entries[names_index]
    names = data[names_entry[4]:names_entry[4] + names_entry[5]]
    result = {}
    for entry in entries:
        name = names[entry[0]:].split(b'\0', 1)[0].decode()
        if entry[1] != 8:  # SHT_NOBITS has no bytes in the ELF file.
            begin, end = entry[4], entry[4] + entry[5]
            if end > len(data):
                raise ValueError('ELF section extends past file')
            result[name] = (entry[3], data[begin:end])
    return result


def payloads(path):
    data = path.read_bytes()
    region = sections(data)
    output = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True)
    symbols = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    result = {}
    for name, address in symbols.items():
        if not name.startswith('_binary_') or not name.endswith('_start'):
            continue
        end = symbols.get(name.removesuffix('_start') + '_end')
        if end is None or end <= address:
            raise ValueError('Invalid embedded ELF symbol bounds: ' + name)
        found = [blob[address - base:end - base] for base, blob in region.values()
                 if base <= address < end <= base + len(blob)]
        if len(found) != 1:
            raise ValueError('Embedded ELF must occupy one unambiguous section: ' + name)
        blob = found[0]
        tpc_sections = sections(blob)
        if '.text' not in tpc_sections:
            raise ValueError('Embedded TPC ELF has no .text: ' + name)
        result[name.removesuffix('_start')] = {'elf': blob, 'text': tpc_sections['.text'][1]}
    if not result:
        raise ValueError('No independently accessible embedded TPC ELFs: ' + str(path))
    return result


def guids(path):
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    function = library.GetKernelGuids
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(GuidInfo)]
    function.restype = ctypes.c_int
    count = ctypes.c_uint32(0)
    if function(3, ctypes.byref(count), None) != 0 or not 1 <= count.value <= 65536:
        raise ValueError('Invalid Gaudi2 GUID count')
    storage = (GuidInfo * count.value)()
    capacity = count.value
    if function(3, ctypes.byref(count), storage) != 0 or count.value != capacity:
        raise ValueError('Unstable GUID enumeration')
    names = [item.name.decode() for item in storage]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('Empty/duplicate registered GUID')
    return names


def inventory(paths, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    seen = set()
    for i, path in enumerate(paths):
        path = path.resolve(strict=True)
        names = guids(path)
        duplicates = seen.intersection(names)
        if duplicates:
            raise ValueError('GUID collisions across databases: ' + str(sorted(duplicates)))
        seen.update(names)
        directory = output / f'family-{i:02d}'
        directory.mkdir()
        kernels = []
        for name, payload in payloads(path).items():
            elf = directory / (name + '.elf')
            elf.write_bytes(payload['elf'])
            text = directory / (name + '.text')
            text.write_bytes(payload['text'])
            kernels.append({'symbol': name, 'elf': str(elf.relative_to(output)),
                'elf_bytes': len(payload['elf']), 'elf_sha256': sha(payload['elf']),
                'text_bytes': len(payload['text']), 'text_sha256': sha(payload['text'])})
        rows.append({'path': str(path), 'sha256': sha(path.read_bytes()),
                     'registered_guids': names, 'payloads': kernels})
    result = {'libraries': rows, 'library_count': len(rows), 'registered_guid_count': len(seen),
              'scope': 'Host metadata and embedded ISA only; execution requires trace confirmation'}
    (output / 'registry.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libraries', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = inventory(a.libraries, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'libraries'}, indent=2))

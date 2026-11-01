"""Recognize an exact retained ELF with only GNU_STACK execute permission removed."""
import hashlib
import struct


def matches_noexecstack_variant(raw, expected_sha256):
    if len(raw) < 64 or raw[:6] != b'\x7fELF\x02\x01':
        return False
    offset, = struct.unpack_from('<Q', raw, 32)
    size, count = struct.unpack_from('<HH', raw, 54)
    if size != 56 or offset < 64 or offset + size * count > len(raw):
        return False
    stacks = [offset + i * size for i in range(count)
              if struct.unpack_from('<I', raw, offset + i * size)[0] == 0x6474e551]
    if len(stacks) != 1:
        return False
    at = stacks[0]
    flags, = struct.unpack_from('<I', raw, at + 4)
    filesz, memsz = struct.unpack_from('<QQ', raw, at + 32)
    if flags != 6 or filesz or memsz:
        return False
    # Reconstruct only for hashing, never enable execution or rewrite the file.
    retained = bytearray(raw)
    struct.pack_into('<I', retained, at + 4, 7)
    return hashlib.sha256(retained).hexdigest() == expected_sha256

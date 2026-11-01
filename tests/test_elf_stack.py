import hashlib
import struct
import unittest

from gaudi_kernels.engine.elf_stack import matches_noexecstack_variant
from tools.port_noexecstack_assets import nonexecuting_stack


def image():
    raw = bytearray(160)
    raw[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', raw, 32, 64)
    struct.pack_into('<HH', raw, 54, 56, 1)
    struct.pack_into('<II', raw, 64, 0x6474e551, 7)
    raw[120:] = b'TPC and host code must remain unchanged!!'
    return bytes(raw)


class ELFStackTest(unittest.TestCase):
    def test_exact_metadata_variant_only(self):
        original = image()
        migrated, patches = nonexecuting_stack(original)
        expected = hashlib.sha256(original).hexdigest()
        self.assertTrue(matches_noexecstack_variant(migrated, expected))
        self.assertEqual(patches, [dict(offset=68, before=7, after=6)])
        self.assertEqual(nonexecuting_stack(migrated), (migrated, []))
        tampered = bytearray(migrated)
        tampered[-1] ^= 1
        self.assertFalse(matches_noexecstack_variant(tampered, expected))
        self.assertFalse(matches_noexecstack_variant(original, expected))

    def test_truncated_headers_and_stack_payload_cannot_qualify(self):
        original = image()
        expected = hashlib.sha256(original).hexdigest()
        for raw in (b'not an ELF', original[:80]):
            self.assertFalse(matches_noexecstack_variant(raw, expected))
        bad = bytearray(original)
        struct.pack_into('<Q', bad, 96, 1)
        with self.assertRaisesRegex(ValueError, 'payload'):
            nonexecuting_stack(bad)
        self.assertFalse(matches_noexecstack_variant(bad, expected))

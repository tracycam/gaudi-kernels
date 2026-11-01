import unittest

from tools.validation.nic_snapshot import rdma_to_hardware


class NicPortMappingTests(unittest.TestCase):
    def test_all_ports(self):
        mapping = rdma_to_hardware(0xffffff, list(range(1, 25)))
        self.assertEqual(mapping[9], 8)
        self.assertEqual(mapping[10], 9)

    def test_external_ports_disabled_preserves_internal_nine(self):
        mapping = rdma_to_hardware(0x3ffeff, list(range(1, 22)))
        self.assertEqual(mapping[9], 9)
        self.assertEqual(mapping[21], 21)
        self.assertTrue({8, 22, 23}.isdisjoint(mapping.values()))

    def test_other_holes(self):
        self.assertEqual(rdma_to_hardware(0b101010, [1, 2, 3]), {1: 1, 2: 3, 3: 5})

    def test_inconsistent_snapshot_is_rejected(self):
        for ports in ([1], [1, 2, 3], [0, 1], [1, 1]):
            with self.subTest(ports=ports), self.assertRaises(ValueError):
                rdma_to_hardware(0b101, ports)

    def test_empty_and_negative_masks(self):
        self.assertEqual(rdma_to_hardware(0, []), {})
        with self.assertRaises(ValueError):
            rdma_to_hardware(-1, [])


if __name__ == '__main__':
    unittest.main()

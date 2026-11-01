import unittest
import numpy as np

from gaudi_kernels.serving.executor.packing import unpack_acc32_lanes


class Acc32LayoutTests(unittest.TestCase):
    def test_each_bf16_vector_stores_even_then_odd_accumulator_lanes(self):
        rows = np.arange(3*512, dtype=np.float32).reshape(3,512)
        stored = rows.reshape(3,4,64,2).swapaxes(-1,-2).reshape(3,512)
        # Endpoints happen to be fixed; an endpoints-only oracle misses this.
        self.assertEqual(stored[0,1], 2)
        self.assertEqual(stored[0,64], 1)
        np.testing.assert_array_equal(unpack_acc32_lanes(stored), rows)

    def test_rejects_non_accumulator_storage(self):
        for a in (np.zeros(127,np.float32),np.zeros(128,np.float16),np.float32(1)):
            with self.assertRaises(ValueError):
                unpack_acc32_lanes(a)

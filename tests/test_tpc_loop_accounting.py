import unittest
from tools.consolidation.analyze_tpc_loop import analyze


class TPCLoopAccountingTests(unittest.TestCase):
    def test_loop_delay_not_repeated_and_symbols_not_used(self):
        text='''0: loop 0, S2, 1, <, 128, SP1
20: nop; nop; nop; nop
40: nop; add.i32 S0, S0, 1; nop; nop // compressed, part1
50: nop; nop; nop; nop // compressed, part2
0000060 .LBB_stale_label:
60: nop; nop; mac.bf16 acc_fp32 D0, V2, S3; nop
80: nop; nop; mac.f32 V0, V1, V2; nop
a0: nop; nop; nop; nop'''
        result=analyze(text,0)
        self.assertEqual(result['repeated_decoded_packets'],4)
        self.assertEqual(result['first_repeated_instruction'],'0x40')
        self.assertEqual(result['slot_non_nop'],[0,1,2,0])
        with self.assertRaises(ValueError):analyze(text,1)

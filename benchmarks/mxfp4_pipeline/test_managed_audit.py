"""Fault-injection gates using a real compiler-sliced M512 graph."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from audit_managed import audit

CASE = Path(__file__).resolve().parents[2] / 'artifacts/builds/mxfp4-reuse/initial/results/gp-m512-auto-a'


class ManagedAudit(unittest.TestCase):
    def test_real_graph_and_faults(self):
        self.assertTrue(audit(CASE)['pass_'])
        original = json.loads((CASE/'post_graph.json').read_text())
        failures = []
        for fault in ('duplicate_decode', 'overlap_alias', 'expanded_dram', 'wrong_mme_b', 'wrong_accumulator'):
            data = copy.deepcopy(original)
            graph = data['graphs'][0]
            ts = {t['name']:t for t in graph['tensors']}
            dec = [n for n in graph['nodes'] if n['guid']=='gk_mxfp4_decode_bf16_v1']
            mme = [n for n in graph['nodes'] if n['engine']=='MME']
            if fault == 'duplicate_decode':
                graph['nodes'].append(copy.deepcopy(dec[0]))
            elif fault == 'overlap_alias':
                ts[dec[1]['input_tensors'][0]]['offset'] = ts[dec[0]['input_tensors'][0]]['offset']
            elif fault == 'expanded_dram':
                ts[dec[0]['output_tensors'][0]]['allocation'] = 'DRAM'
            elif fault == 'wrong_mme_b':
                mme[0]['input_tensors'][1] = mme[0]['input_tensors'][0]
            else:
                ts[mme[0]['output_tensors'][0]]['dtype'] = 'bf16'
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for name in ('run.log','exit.json'):
                    (root/name).write_bytes((CASE/name).read_bytes())
                (root/'post_graph.json').write_text(json.dumps(data))
                with self.subTest(fault=fault), self.assertRaises(AssertionError):
                    audit(root)
                failures.append(fault)
        self.assertEqual(len(failures),5)


if __name__ == '__main__':
    unittest.main()

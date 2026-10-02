from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os

from gaudi_kernels.engine.config import ConfigError, EngineConfig, HostPlacement
from gaudi_kernels.serving.host_placement import discover, install_worker_spawn


class PlacementTests(unittest.TestCase):
    def test_module_order_not_device_index_and_cgroup_mask_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for module in range(8):
                path = root/f'0000:{module:02x}:00.0'; path.mkdir()
                (path/'numa_node').write_text(str(module//4))
                (path/'local_cpulist').write_text('0-7' if module<4 else '8-15')
            query = '\n'.join(f'{module}, 0000:{module:02x}:00.0' for module in reversed(range(8)))
            with patch('os.sched_getaffinity', return_value={2,3,10,11}):
                plan = discover(HostPlacement('local'), pci_root=root, query=query)
            self.assertEqual([w['numa_node'] for w in plan['workers']], [0]*4+[1]*4)
            self.assertEqual(plan['workers'][0]['allowed_local_cpus'], [2,3])
            self.assertEqual(plan['workers'][4]['allowed_local_cpus'], [10,11])

    def test_host_placement_does_not_change_arithmetic_fingerprint(self):
        old = EngineConfig()
        new = EngineConfig.from_dict({'runtime': {'host': {'numa_binding': 'local'}}})
        self.assertEqual(old.arithmetic_fingerprint, new.arithmetic_fingerprint)
        with self.assertRaises(ConfigError):
            EngineConfig.from_dict({'runtime': {'host': {'visible_modules': [0]*8}}})

    def test_module_selected_before_child_start_and_parent_environment_restored(self):
        observed = []
        class Worker:
            @staticmethod
            def make_worker_process(vllm_config, local_rank):
                observed.append(os.environ['HLS_MODULE_ID'])
                if local_rank == 1:
                    raise RuntimeError('spawn fixture failure')
        module = SimpleNamespace(WorkerProc=Worker)
        runtime = SimpleNamespace(selection=SimpleNamespace(engine=EngineConfig()))
        with patch('gaudi_kernels.engine.context.context', return_value=runtime), patch.dict(os.environ, {'HLS_MODULE_ID': 'parent'}):
            install_worker_spawn(module)
            module.WorkerProc.make_worker_process(None, 0)
            with self.assertRaises(RuntimeError):
                module.WorkerProc.make_worker_process(None, 1)
            self.assertEqual(observed, ['0', '1'])
            self.assertEqual(os.environ['HLS_MODULE_ID'], 'parent')

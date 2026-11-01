"""CPU policy ownership: restore on partial failure and include new threads."""
import errno
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from gaudi_kernels.engine.config import ConfigError, EngineConfig
from gaudi_kernels.serving.host_qos import WorkerQos


class HostQosTest(unittest.TestCase):
    def test_existing_cap_survives_partial_startup_failure(self):
        attrs = {11: (0, 1024), 22: (12, 512)}

        def read(tid):
            low, high = attrs[tid]
            return SimpleNamespace(util_min=low, util_max=high)

        def write(tid, low, high):
            attrs[tid] = (low, high)

        with (patch('gaudi_kernels.serving.host_qos.os.getpid', return_value=11),
              patch('gaudi_kernels.serving.host_qos.thread_ids', return_value=[11, 22]),
              patch('gaudi_kernels.serving.host_qos.read_attr', side_effect=read),
              patch('gaudi_kernels.serving.host_qos.set_clamp', side_effect=write)):
            with self.assertRaisesRegex(ValueError, 'existing CPU utilization cap'):
                WorkerQos(1024)
        self.assertEqual(attrs, {11: (0, 1024), 22: (12, 512)})

    def test_inherited_thread_and_initial_settings_are_restored_once(self):
        tids = [11, 22]
        attrs = {11: (0, 1024), 22: (12, 1024)}

        def read(tid):
            low, high = attrs[tid]
            return SimpleNamespace(util_min=low, util_max=high)

        def write(tid, low, high):
            attrs[tid] = (low, high)

        with (patch('gaudi_kernels.serving.host_qos.os.getpid', return_value=11),
              patch('gaudi_kernels.serving.host_qos.thread_ids', side_effect=lambda: tids),
              patch('gaudi_kernels.serving.host_qos.read_attr', side_effect=read),
              patch('gaudi_kernels.serving.host_qos.set_clamp', side_effect=write) as setter):
            qos = WorkerQos(1024)
            tids.append(33)
            attrs[33] = (1024, 1024)
            qos.restore()
            calls = setter.call_count
            qos.restore()
            self.assertEqual(setter.call_count, calls)
        self.assertEqual(attrs, {11: (0, 1024), 22: (12, 1024), 33: (0, 1024)})

    def test_manifest_rejects_invalid_minimum_before_worker_creation(self):
        with self.assertRaises(ConfigError):
            EngineConfig.from_dict({'runtime': {'host': {'worker_util_min': 1025}}})
        self.assertEqual(EngineConfig.from_dict({}).runtime.host.worker_util_min, 1024)
        self.assertEqual(EngineConfig.from_dict({'runtime': {'host': {'worker_util_min': 0}}}).runtime.host.worker_util_min, 0)

    def test_refresh_clamps_post_startup_threads_and_restores_originals(self):
        tids = [11, 22]
        attrs = {11: (0, 1024), 22: (12, 1024)}

        def read(tid):
            low, high = attrs[tid]
            return SimpleNamespace(util_min=low, util_max=high)

        def write(tid, low, high):
            attrs[tid] = (low, high)

        with (patch('gaudi_kernels.serving.host_qos.os.getpid', return_value=11),
              patch('gaudi_kernels.serving.host_qos.thread_ids', side_effect=lambda: tids),
              patch('gaudi_kernels.serving.host_qos.read_attr', side_effect=read),
              patch('gaudi_kernels.serving.host_qos.set_clamp', side_effect=write)):
            qos = WorkerQos(1024)
            tids.append(44)
            attrs[44] = (64, 1024)
            self.assertEqual(qos.refresh(), 3)
            self.assertEqual(attrs[44], (1024, 1024))
            qos.restore()
        self.assertEqual(attrs, {11: (0, 1024), 22: (12, 1024), 44: (64, 1024)})

    def test_primary_physical_cpus_strips_smt_siblings(self):
        import tempfile
        from pathlib import Path
        from gaudi_kernels.serving.host_qos import primary_physical_cpus
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cpu, sib in ((0, '0,56'), (1, '1,57'), (56, '0,56'), (57, '1,57')):
                p = root / f'cpu{cpu}/topology'
                p.mkdir(parents=True)
                (p / 'thread_siblings_list').write_text(sib)
            self.assertEqual(primary_physical_cpus({0, 1, 56, 57}, sysfs_root=root), {0, 1})


if __name__ == '__main__':
    unittest.main()

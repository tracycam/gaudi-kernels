import ast
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gaudi_kernels.engine.config import ConfigError, EngineConfig
from gaudi_kernels.engine.selection import PolicySelection
from gaudi_kernels.engine import context as runtime
from gaudi_kernels.serving.executor import moe_dispatch_runtime as dispatch


class ServingContextTests(unittest.TestCase):
    def setUp(self):
        self.pins_patch=patch.object(runtime,'PINS',{})
        self.pins_patch.start()
    def tearDown(self):
        runtime._CONTEXT = None
        self.pins_patch.stop()

    def document(self, folder):
        path=Path(folder)/'library.so';path.write_bytes(b'qualification fixture')
        runtime.PINS['native.torch']=hashlib.sha256(path.read_bytes()).hexdigest()
        return {'schema_version':1,'startup':{'run_dir':str(Path(folder)/'run')},
                'artifacts':{'native.torch':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}},
                'selection':PolicySelection(EngineConfig()).to_dict()}

    def test_manifest_checks_hashes_before_initializing(self):
        with tempfile.TemporaryDirectory() as folder:
            document=self.document(folder)
            Path(document['artifacts']['native.torch']['path']).write_bytes(b'wrong bytes')
            with self.assertRaises(ConfigError):runtime.initialize(document,directory=folder)
            with self.assertRaises(ConfigError):runtime.context()

    def test_explicit_bindings_ignore_historical_environment(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{
                'GK_BLOCK_FP8_CORE_VARIANT':'invalid','UNIFIED_COMPACT_MAX_ROWS':'999',
                'UNIFIED_PRECISION_REDUCE':'off'}):
            context=runtime.initialize(self.document(folder),directory=folder)
            self.assertTrue(context.has('native'))
            self.assertEqual(context.collective_mode,'auto')
            dispatch.set_rows((1,8))
            self.assertEqual(dispatch.select_mode('auto',2),'broadcast')
            self.assertEqual(dispatch.select_mode('auto',8),'compact')
            with self.assertRaises(ConfigError):runtime.initialize(self.document(folder),directory=folder)

    def test_layer_count_is_distinct_from_sdk_graph_flush_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            document=self.document(folder);document['startup']['layers']=2
            ctx=runtime.initialize(document,directory=folder)
            self.assertEqual(ctx.startup.layers,2)
            self.assertEqual(ctx.selection.engine.runtime.graph_layer_interval,70)
        with self.assertRaises(ConfigError):EngineConfig.from_dict({'runtime':{'graph_layer_interval':2}})

    def test_shared_tpc_database_does_not_require_a_same_name_torch_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            document=self.document(folder)
            binding=document['artifacts'].pop('native.torch')
            runtime.PINS['swa_bundle.tpc']=binding['sha256']
            document['artifacts']['swa_bundle.tpc']=binding
            ctx=runtime.initialize(document,directory=folder)
            self.assertTrue(ctx.has('swa_bundle'))
            self.assertEqual(str(ctx.artifact('swa_bundle','tpc').path),binding['path'])
            self.assertFalse(ctx.has('absent'))

    def test_moe_sidecars_are_pinned_and_colocated_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            document=self.document(folder)
            for key,name in (('moe_bundle.tpc','bundle.so'),
                             ('moe_batch_provider.host','provider_batch.so'),
                             ('moe_expert_provider.host','provider_expert.so')):
                path=Path(folder)/name;path.write_bytes(key.encode())
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                runtime.PINS[key]=digest
                document['artifacts'][key]={'path':str(path),'sha256':digest}
            missing=document['artifacts'].pop('moe_batch_provider.host')
            with self.assertRaisesRegex(ConfigError,'provider'):
                runtime.initialize(document,directory=folder)
            document['artifacts']['moe_batch_provider.host']=missing
            runtime.initialize(document,directory=folder)
            runtime._CONTEXT=None
            (Path(folder)/'provider_expert.so').write_bytes(b'tampered')
            with self.assertRaisesRegex(ConfigError,'bytes changed'):
                runtime.initialize(document,directory=folder)

    def test_missing_adapter_boundary_is_rejected_before_capture(self):
        from gaudi_kernels.serving.model_adapter import adapter_class
        with self.assertRaisesRegex(RuntimeError, 'position boundary'):
            adapter_class(type('OldAdapter', (), {}))

    def test_production_boot_has_no_benchmark_startup_dependency(self):
        from gaudi_kernels.serving import launch
        source=Path(launch.__file__).read_text()
        self.assertNotIn('tools/validation',source)
        self.assertNotIn('benchmark-module',source)

    def test_framework_boot_does_not_require_executor_artifacts(self):
        from gaudi_kernels.serving import bootstrap
        from gaudi_kernels.serving.executor import native_backend
        with tempfile.TemporaryDirectory() as folder:
            document = self.document(folder)
            document['selection']['engine']['runtime']['executor'] = 'pytorch'
            ctx = runtime.initialize(document, directory=folder)
            with patch.object(bootstrap, 'load_manifest', return_value=ctx), \
                 patch.dict('os.environ', {'GK_RUNTIME_MANIFEST': 'fixture.json'}), \
                 patch.dict('sys.modules', {'gaudi_kernels.serving.bootstrap_hooks': SimpleNamespace()}), \
                 patch.object(bootstrap.ctypes, 'CDLL', side_effect=AssertionError('external executor loaded')):
                self.assertIs(bootstrap.start(), ctx)
                # install must return before importing Torch or probing e1 ABI.
                with patch.dict('sys.modules', {'torch': None}):
                    native_backend.install(SimpleNamespace())
                with self.assertRaisesRegex(ValueError, 'absent'):
                    native_backend.configure(SimpleNamespace(), True)

    def test_framework_launcher_keeps_kernels_without_preload(self):
        from gaudi_kernels.serving import launch, host_placement
        with tempfile.TemporaryDirectory() as folder:
            document = self.document(folder)
            document['selection']['engine']['runtime']['executor'] = 'pytorch'
            ctx = runtime.initialize(document, directory=folder)
            manifest = Path(folder)/'manifest.json'
            manifest.write_text('{}')
            placement = {'binding': 'none', 'workers': [{'module_id': i} for i in range(8)]}
            with patch.object(launch, 'load_manifest', return_value=ctx), \
                 patch.object(host_placement, 'prepare', return_value=placement), \
                 patch.dict('os.environ', {'LD_PRELOAD': '/missing/old-executor.so'}), \
                 patch.object(launch.os, 'execve') as execute:
                launch.launch(manifest=manifest, plugin_source=Path(folder),
                              vllm_source=Path(folder), module='fixture')
            environment = execute.call_args.args[2]
            self.assertNotIn('LD_PRELOAD', environment)
            self.assertIn('GC_KERNEL_PATH', environment)
            self.assertIn('GK_RUNTIME_MANIFEST', environment)

    def test_manifest_is_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            for mutation in ('bool_layers','text_bool','typo','wrong_hash','extra_binding'):
                document=self.document(folder)
                if mutation=='bool_layers':document['startup']['layers']=True
                if mutation=='text_bool':document['startup']['inventory']='false'
                if mutation=='typo':document['startup']['inventroy']=False
                if mutation=='wrong_hash':document['artifacts']['native.torch']['sha256']='G'*64
                if mutation=='extra_binding':document['artifacts']['native.torch']['enable']=True
                with self.subTest(mutation=mutation),self.assertRaises(ConfigError):runtime.initialize(document,directory=folder)

    def test_canonical_source_has_no_legacy_import_or_env_policy(self):
        root=Path(__file__).resolve().parents[1]
        files=list((root/'python/gaudi_kernels/serving').rglob('*.py'))
        files += [root/'python/gaudi_kernels'/name for name in
                  ('production_integration.py','vllm_block_fp8.py','fp32_artifact_binding.py',
                   'norm_grid24_binding.py','block_fp8_fp32_contract.py')]
        for path in files:
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('tools.','experiments.')),(path,node.module))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    name=ast.unparse(node.func)
                    if name in ('os.getenv','os.environ.get'):
                        key=node.args[0]
                        self.assertIsInstance(key,ast.Constant,path)
                        self.assertIn(key.value,('GC_KERNEL_PATH','GK_RUNTIME_MANIFEST','HLS_MODULE_ID'),(path,key.value))
            if 'context()' in path.read_text():
                self.assertTrue(any(isinstance(n,ast.ImportFrom) and any(a.name=='context' for a in n.names)
                                    for n in ast.walk(tree)) or path.name in ('context.py',),path)
        # Model code already uses `context` as a tensor/forward-context local.
        # The settings accessor must stay a distinct global/free symbol.
        import symtable
        def check_symbols(table, path):
            if table.get_type()=='function' and 'execution_context' in table.get_identifiers():
                symbol=table.lookup('execution_context')
                self.assertFalse(symbol.is_assigned() or symbol.is_parameter(),(path,table.get_name()))
            for child in table.get_children():check_symbols(child,path)
        for path in files:check_symbols(symtable.symtable(path.read_text(),str(path),'exec'),path)
        for path in (root/'csrc/legacy_executor').iterdir():
            self.assertNotIn('getenv(',path.read_text(),path)
        self.assertFalse((root/'tools/consolidation/migrate_selection.py').exists())


if __name__=='__main__':unittest.main()

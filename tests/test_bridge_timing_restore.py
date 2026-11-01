import unittest

from gaudi_kernels.serving.diagnostics.bridge_timing import restore_methods


class TimingRestoreTest(unittest.TestCase):
    def test_inherited_method_remains_visible_to_later_class_hooks(self):
        class Runner:
            def execute(self):
                return 'original'

        runner = Runner()
        runner.execute = lambda: 'observer'
        restore_methods({'originals': [(runner, 'execute', False, None)]})
        self.assertNotIn('execute', vars(runner))
        Runner.execute = lambda self: 'quality hook'
        self.assertEqual(runner.execute(), 'quality hook')

    def test_existing_instance_override_is_restored(self):
        class Runner:
            pass

        runner = Runner()
        original = lambda: 'instance override'
        runner.execute = lambda: 'observer'
        restore_methods({'originals': [(runner, 'execute', True, original)]})
        self.assertIs(runner.execute, original)


if __name__ == '__main__':
    unittest.main()

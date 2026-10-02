import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.validation.executor.batch_prompts import load


class BatchPromptTests(unittest.TestCase):
    def test_original_tokens_and_identity_survive_without_answer_rubric(self):
        ids=[9,8,7]
        row=dict(id='business-task',prompt_token_ids=ids,answer_checks=['never an input'],
                 prompt_token_ids_sha256=hashlib.sha256(b'[9,8,7]').hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'corpus.jsonl';path.write_text(json.dumps(row)+'\n')
            prompts,identity=load(path,needed=1,maximum_length=3)
            self.assertEqual(prompts,[dict(prompt_token_ids=ids)])
            self.assertEqual(identity['requests'][0]['length'],3)
            with self.assertRaisesRegex(ValueError,'truncation'):
                load(path,needed=1,maximum_length=2)
            row['prompt_token_ids'][1]=6;path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                load(path,needed=1,maximum_length=3)

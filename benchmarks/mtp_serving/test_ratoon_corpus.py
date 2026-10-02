"""CPU regressions for tokenizer return types at the corpus/harness boundary."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transformers import BatchEncoding

import ratoon_corpus as corpus


class FakeTokenizer:
    """Same prompt IDs, different real tokenizer API return containers."""

    def __init__(self, representation):
        self.representation = representation

    @staticmethod
    def ids(messages):
        return [101] + [ord(c) for m in messages for c in m["role"] + m["content"]] + [102]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        ids = self.ids(messages)
        if self.representation == "list":
            return ids
        if self.representation == "mapping":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.representation == "batch_encoding":
            return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
        return BatchEncoding({"input_ids": [ids], "attention_mask": [[1] * len(ids)]})

    def get_chat_template(self, **kwargs):
        return "synthetic CPU test template"


class TokenizerReturnContract(unittest.TestCase):
    def test_selection_counts_and_exports_identical_ids_for_all_containers(self):
        with tempfile.TemporaryDirectory(prefix="ratoon-tokenizer-contract-") as name:
            root = Path(name)
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            (tokenizer_path / "tokenizer_config.json").write_text("{}\n")
            source = dict(path="module.ts", sha256="source", repository="component",
                          repository_path="module.ts", git_revision="revision", matches_revision=True,
                          original_path="/source/module.ts", resolved_path="/source/module.ts")
            core = corpus.block(source, 1, 1, ["const stable = 1;\n"])
            optional = corpus.block(source, 2, 3, ["const stable = 1;\n", "const extra = 2;\n",
                                                   "const tooLong = '" + "x" * 1000 + "';\n"])
            sample = dict(id="return-contract", task="Explain stable and extra.", answer_checks=["rubric"],
                          scope="synthetic test", expansion_reason="test", core=[core], optional=[optional])
            partial = {**optional, "end_line": 2, "content": "const extra = 2;\n"}
            budget = len(FakeTokenizer.ids(corpus.render(sample, [core, partial])))
            core_size = len(FakeTokenizer.ids(corpus.render(sample, [core])))
            frozen = root / "corpus.json"
            corpus.dump(frozen, dict(source_root=str(root / "source"), sources={"module.ts": source},
                                     samples=[sample]))
            exported = []
            for representation in ("list", "mapping", "batch_encoding", "singleton_batch"):
                with self.subTest(representation=representation):
                    args = Namespace(corpus=str(frozen), output=str(root / representation), unit="tokens",
                                     tokenizer=str(tokenizer_path), chat_template_kwargs='{"enable_thinking":false}',
                                     ids=None, budgets=str(budget), include_token_ids=True)
                    fake = FakeTokenizer(representation)
                    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake):
                        count, _, encode = corpus.counter(args)
                        self.assertEqual(count(corpus.render(sample, [core])), core_size)
                        # A Mapping has two keys, but thousands of token IDs.
                        self.assertGreater(core_size, 2)
                        with self.assertRaisesRegex(ValueError, "required core costs"):
                            corpus.prefix_fit(sample, core_size - 1, count)
                        with contextlib.redirect_stdout(io.StringIO()):
                            corpus.select(args)
                        row = json.loads((Path(args.output) / f"tokens-{budget}.jsonl").read_text())
                        ids = row["prompt_token_ids"]
                        self.assertIs(type(ids), list)
                        self.assertTrue(all(type(token) is int for token in ids))
                        self.assertEqual(ids, fake.ids(row["messages"]))
                        self.assertEqual(ids, encode(row["messages"]))
                        self.assertEqual(len(ids), row["length"]["prompt_tokens"])
                        self.assertEqual(len(ids), row["length"]["measured"])
                        self.assertLessEqual(len(ids), budget)
                        self.assertEqual(row["sources"][-1]["end_line"], 2)
                        self.assertNotIn("tooLong", row["messages"][-1]["content"])
                        self.assertEqual(row["prompt_token_ids_sha256"], corpus.digest(
                            json.dumps(ids, separators=(",", ":")).encode()))
                        exported.append(ids)
            self.assertTrue(all(ids == exported[0] for ids in exported))

    def test_ambiguous_or_malformed_tokenizer_outputs_fail(self):
        for encoded in ({"attention_mask": [1, 1]}, [[1, 2], [3, 4]], [True, 2], [1.5, 2], "12"):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                corpus.single_token_ids(encoded)


if __name__ == "__main__":
    unittest.main()

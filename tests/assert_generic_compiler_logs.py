"""Check fresh GraphCompiler logs after the generic norm and attention probes."""

import argparse
import os
from pathlib import Path
import re
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+", type=Path)
args = parser.parse_args()
problems = re.compile(
    r"duplicated dims|No Entry point GetSupportedDataLayouts|"
    r"GetSupportedLayouts pointer is null for kernel gk_",
    re.IGNORECASE,
)
for path in args.logs:
    content = path.read_text()
    if not content:
        raise AssertionError(f"empty GraphCompiler log: {path}")
    violations = [line for line in content.splitlines() if problems.search(line)]
    if violations:
        raise AssertionError(f"GraphCompiler violations in {path}:\n" + "\n".join(violations))
assert "habana_frameworks" not in sys.modules
print(f"GraphCompiler access-axis and provider-layout warnings absent from {len(args.logs)} log(s)")

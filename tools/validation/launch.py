"""Benchmark-only launcher; historical control RPCs never enter serving boot."""
import argparse
from pathlib import Path
import sys

from gaudi_kernels.serving.launch import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plugin-source', type=Path, required=True)
    parser.add_argument('--vllm-source', type=Path, required=True)
    parser.add_argument('--benchmark-module', required=True)
    parser.add_argument('--python', default=sys.executable)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == '--':
        remainder = remainder[1:]
    launch(manifest=args.manifest, plugin_source=args.plugin_source, vllm_source=args.vllm_source,
           module=args.benchmark_module, python=args.python, arguments=remainder,
           bootstrap=Path(__file__).resolve().parent / 'startup', output_argument=True)


if __name__ == '__main__':
    main()

"""Keep all environment changes within one explicit-module probe child."""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--buffers', type=int, choices=[1, 2], required=True)
p.add_argument('--audit', action='store_true')
p.add_argument('--profile', action='store_true')
p.add_argument('--ordered', action='store_true')
p.add_argument('--hard-control', action='store_true')
p.add_argument('--compiler-sram', action='store_true',
               help='Diagnostic transient weights; actual SRAM placement must be audited')
p.add_argument('--logical-weight-mib',type=int)
p.add_argument('--compile-only',action='store_true')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[0] == '--' else a.command
os.environ['GK_PIPELINE_BUFFERS'] = str(a.buffers)
os.environ.pop('GK_PIPELINE_ORDERED', None)
os.environ.pop('GK_PIPELINE_COMPILER_SRAM', None)
os.environ.pop('GK_MXFP4_LOGICAL_WEIGHT_MIB', None)
os.environ.pop('GK_MXFP4_COMPILE_ONLY', None)
if a.logical_weight_mib is not None:
    if not a.compiler_sram or not 1 <= a.logical_weight_mib <= 512:
        p.error('logical weight size requires compiler-managed mode and 1..512 MiB')
    os.environ['GK_MXFP4_LOGICAL_WEIGHT_MIB']=str(a.logical_weight_mib)
if a.compile_only:
    os.environ['GK_MXFP4_COMPILE_ONLY']='1'
if a.compiler_sram:
    if a.buffers != 1 or a.ordered:
        p.error('--compiler-sram requires --buffers 1 and no --ordered')
    os.environ['GK_PIPELINE_COMPILER_SRAM'] = '1'
if a.ordered:
    os.environ['GK_PIPELINE_ORDERED'] = '1'
if a.hard_control:
    os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', MAKE_CTRL_DEP_SOFT='false')
if a.profile:
    out = Path(os.environ['PROBE_OUT'])
    trace = out / 'trace'
    trace.mkdir()
    config = out / 'profile.json'
    subprocess.run(['hl-prof-config', '--gaudi2', '--config-filename', str(config),
                    '-e', 'off', '-o', str(trace), '-s', 'pipeline', '--phase',
                    'multi-enq', '-g', '7-7', '-b', '64', '--invoc', 'json',
                    '--merged', 'json,hltv', '--trace-analyzer', 'on',
                    '--trace-analyzer-csv', 'on', '--host', 'on', '--add-pid',
                    'off'], check=True)
    os.environ.update(HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(config))
if a.audit:
    out = Path(os.environ['PROBE_OUT'])
    (out / 'graphs').mkdir()
    os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', GRAPH_VISUALIZATION='1',
                      GRAPH_VISUALIZATION_DIR=str(out / 'graphs'),
                      SRAM_SLICER_GRAPH_VISUALIZATION='1',
                      DUMP_POST_GRAPHS=str(out / 'post_graph.json'))
os.execv(command[0], command)

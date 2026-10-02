"""Local orchestration, one fresh process per same-GUID arm, no nested runner."""
import argparse,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=22);p.add_argument('--remote-root',required=True);p.add_argument('--socket',required=True);p.add_argument('--python',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];commit=json.loads((root/'source-identity.json').read_text())['git_commit'];records=[]
for index,arm in enumerate(('control','hoist','hoist','control')):
 case='hoist-abba-'+str(index)+'-'+arm;core='libraries/libmxfp4_moe_graph_tpc.so'if arm=='control'else'builds/core-hoist/libmxfp4_moe_graph_tpc.so';tiles='libraries/libgaudi_route_tiles_tpc.so'if arm=='control'else'builds/tiles-hoist/libgaudi_route_tiles_tpc.so'
 cmd=[a.python,'tools/run_device_probe.py','--module','7','--case',case,'--git-commit',commit,'--timeout','300']
 for library in (core,'libraries/libgaudi_route_metadata_v2_tpc.so',tiles):cmd+=['--kernel-library',library]
 cmd+=['--',a.python,'benchmarks/route_tile_hoist/device.py','--arm',arm,'--fixture','fixtures/t513-e384','--core-library','libraries/gaudi_mxfp4_moe_graph.so','--metadata-library','libraries/gaudi_route_metadata_v2.so','--tiles-library','libraries/gaudi_route_tiles.so','--rows','32','--n-tile','2048','--replays','4']
 audit=[a.python,'benchmarks/moe_route_tiles/audit.py','results/'+case]
 command=['ssh','-S',a.socket,'-o','BatchMode=yes','-p',str(a.port),a.host,'cd '+shlex.quote(a.remote_root)+' && '+shlex.join(cmd)+' && '+shlex.join(audit)]
 begin=time.perf_counter()
 with(a.output/(case+'.log')).open('w')as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=360)
 row=dict(case=case,arm=arm,source_commit=commit,command=command,exit=result.returncode,wall_s=time.perf_counter()-begin);records.append(row);(a.output/'driver.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(row),flush=True)
 if result.returncode:raise SystemExit(result.returncode)

"""Read-only bounded remote progress monitor; seal completed owned cases locally.

Never starts, retries, kills or resets remote work. Missing SSH or a reboot is
not completion. The full result remains the authority; this writes live views.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('case', 'host', 'control', 'remote-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=4000)
    p.add_argument('--interval', type=int, default=45)
    p.add_argument('--runner-input', type=Path, action='append', default=[])
    a = p.parse_args()
    if not 1 <= a.interval <= 60 or not 60 <= a.timeout <= 7200:
        p.error('bounded interval 1..60 and timeout 60..7200 required')
    if not a.case or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.case):
        p.error('single safe case name required')
    ssh = ['ssh', '-S', a.control, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p', str(a.port), a.host, 'python3 -']
    script = '''from pathlib import Path
import json
r=Path(REMOTE);c=CASE
out={'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name,key in [(c+'.exit.json','runner'),(c+'/result.json','result')]:
 p=r/name
 if not p.exists():continue
 try:x=json.loads(p.read_text())
 except Exception as e:out[key+'_error']=str(e);continue
 if key=='runner':out[key]=x;continue
 out[key]={k:x.get(k)for k in ['status','candidate_accepted','init_s','fp32_quality','chat_quality','batch_policy_abba'] if k in x and k!='batch_policy_abba'}
 out[key]['runs']=[{k:v.get(k)for k in ['name','policy','execution_gate_pass','performance_qualified','timing','diagnostic_timing']}for v in x.get('runs',[])]
 out[key]['teacher_rows']=len(x.get('teacher_forced',{}).get('rows',[]))
 out[key]['batch_rows']=len(x.get('batch_runs',[]))
 if 'batch_policy_abba'in x:out[key]['batch_status']=x['batch_policy_abba']['status'];out[key]['batch_comparisons']=x['batch_policy_abba'].get('comparisons',[])
print(json.dumps(out))
'''.replace('REMOTE', repr(a.remote_root)).replace('CASE', repr(a.case))
    live = a.out / (a.case+'-live')
    live.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    while time.monotonic()-begin < a.timeout:
        try:
            response = subprocess.run(ssh, input=script.encode(), stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, timeout=20, check=True)
            state = json.loads(response.stdout)
        except (subprocess.SubprocessError, ValueError) as e:
            state = {'connection_error':str(e), 'status':'UNOBSERVABLE'}
        state['observed_unix_s'] = time.time()
        (live/'latest.json').write_text(json.dumps(state, indent=2)+'\n')
        with (live/'observations.jsonl').open('a') as file:
            file.write(json.dumps(state)+'\n')
        result = state.get('result', {})
        runner = state.get('runner', {})
        print(json.dumps(dict(status=result.get('status',state.get('status')), init_s=result.get('init_s'),
            runs=len(result.get('runs',[])), teacher_rows=result.get('teacher_rows'),
            batch_rows=result.get('batch_rows'), runner_stage=runner.get('stage'),
            elapsed_s=round(time.monotonic()-begin,1))), flush=True)
        recorded_boot = runner.get('runner_process',{}).get('boot_id')
        if recorded_boot and recorded_boot != state.get('boot'):
            raise SystemExit('Observed host reboot; preserve partial case with a separate recovery classification')
        if (runner.get('returncode') is not None and runner.get('cleanup',{}).get('reaped')
                and not runner.get('remaining_owned_processes')):
            command = [sys.executable, str(Path(__file__).with_name('seal_model_case.py'))]
            for key in ('case','host','control','remote_root','port','out'):
                command += ['--'+key.replace('_','-'), str(getattr(a,key))]
            for path in a.runner_input:
                command += ['--runner-input',str(path)]
            subprocess.run(command, check=True)
            return
        time.sleep(a.interval)
    raise SystemExit('Monitor budget ended; remote work was not modified')


if __name__ == '__main__':
    main()

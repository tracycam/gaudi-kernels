"""Small vendor-only health probe after an externally observed host reboot.

Run only through tools/run_device_probe.py. No custom libraries, resets, clock
changes or firmware changes. Passing is not a stress test or model acceptance.
"""
import json
import os
from pathlib import Path
import torch
import habana_frameworks.torch.core as ht

out = Path(os.environ['KERNEL_PROBE_OUT'])
report = dict(status='STARTED', boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              module=int(os.environ['GAUDI_KERNELS_MODULE_ID']), torch=str(torch.__version__),
              custom_kernels=False, scope='small vendor allocation/copy/MME check, not sustained-load health or performance')
(out/'health.json').write_text(json.dumps(report, indent=2)+'\n')
x = torch.ones((2048, 2048), dtype=torch.bfloat16, device='hpu')
w = torch.full_like(x, .5)
ht.mark_step()
torch.hpu.synchronize()
checks = []
for value in (1., 2., -1., 0.):
    x.fill_(value)
    y = x @ w
    ht.mark_step()
    torch.hpu.synchronize()
    actual = y.cpu()
    passed = bool(torch.all(actual == value*1024).item())
    checks.append(dict(value=value, expected=value*1024, checked_values=actual.numel(), passed=passed))
    report['checks'] = checks
    (out/'health.json').write_text(json.dumps(report, indent=2)+'\n')
    assert passed, 'vendor-only MME health check failed'
report['status'] = 'PASS_SMALL_VENDOR_ONLY_PROBE'
(out/'health.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))

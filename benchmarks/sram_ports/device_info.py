"""Read-only attributes; optional module acquisition, no launch or clock change."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser();p.add_argument('--module',type=int,choices=range(8),required=True);p.add_argument('--acquire',action='store_true');a=p.parse_args()
header=Path('/usr/include/habanalabs/synapse_api_types.h').read_text()
enum=re.search(r'typedef enum\s*\{([^}]+)\}\s*synDeviceAttribute',header).group(1)
names=[s.strip() for s in enum.split(',') if s.strip()]
assert names[0]=='DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS' and names[9]=='DEVICE_ATTRIBUTE_MAX_RMW_SIZE'
selected=['DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS','DEVICE_ATTRIBUTE_SRAM_SIZE','DEVICE_ATTRIBUTE_MAX_RMW_SIZE','DEVICE_ATTRIBUTE_CLK_RATE','DEVICE_ATTRIBUTE_TPC_ENABLED_MASK']
api=ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
attrs=(ctypes.c_int*len(selected))(*(names.index(n) for n in selected));values=(ctypes.c_uint64*len(selected))()
api.synDeviceGetAttributeByModuleId.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int),ctypes.c_uint,ctypes.c_uint]
api.synDeviceGetAttribute.argtypes=api.synDeviceGetAttributeByModuleId.argtypes
assert api.synInitialize()==0
dev=ctypes.c_uint();acquired=False
try:
    if a.acquire:
        status=api.synDeviceAcquireByModuleId(ctypes.byref(dev),a.module)
        assert status==0,status
        acquired=True
    attributes={}
    for n in selected:
        attr=ctypes.c_int(names.index(n));value=ctypes.c_uint64()
        query=api.synDeviceGetAttribute if acquired else api.synDeviceGetAttributeByModuleId
        status=query(ctypes.byref(value),ctypes.byref(attr),1,dev.value if acquired else a.module)
        attributes[n]=dict(status=status,value=value.value if status==0 else None)
    result=dict(module=a.module,acquired=acquired,attributes=attributes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hl_smi=subprocess.check_output(['hl-smi','--query-aip=module_id,index,bus_id,memory.used,utilization.aip,clocks.current.soc,clocks.limit.tpc,clocks.limit.soc','--format=csv'],text=True))
    print(json.dumps(result,indent=2))
finally:
    if acquired:assert api.synDeviceRelease(dev.value)==0
    assert api.synDestroy()==0

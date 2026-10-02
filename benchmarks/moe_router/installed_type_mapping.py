"""CPU ABI diagnostic of exact exported enum/bool signatures, no tensor pointers.

Evidence only; production code does not call these private bridge functions.
Run in separate processes for native-int64 0/1 because vendor maps are cached.
"""
import argparse,ctypes,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--native-int64',choices=('0','1'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT=a.native_int64)
import torch
import habana_frameworks.torch as ht
root=Path(ht.__file__).parent/'lib'
plugin=ctypes.CDLL(str(root/'libhabana_pytorch_plugin.so'))
backend=ctypes.CDLL(str(root/'libhabana_pytorch_backend.so'))
enabled=plugin._ZN6common16IsInt64SupportedEv;enabled.argtypes=[];enabled.restype=ctypes.c_bool
# c10::ScalarType is enum class:int8_t; Int=3, Long=4. synDataType is uint32 enum.
mapping=backend._ZN14habana_helpers23pytorch_to_synapse_typeEN3c1010ScalarTypeE
mapping.argtypes=[ctypes.c_int8];mapping.restype=ctypes.c_uint32
i32,i64=mapping(3),mapping(4);assert enabled()==bool(int(a.native_int64))
assert (i32==i64)==(a.native_int64=='0')
r={'status':'CPU_INSTALLED_ENUM_MAPPING_PASS','device_used':False,'native_int64_enabled':enabled(),'Int_synDataType':i32,'Long_synDataType':i64,'same_physical_type':i32==i64,'torch':torch.__version__,'libraries_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('libhabana_pytorch_plugin.so','libhabana_pytorch_backend.so')},'scope':'exact exported scalar-enum/bool ABI diagnostic only; no opaque object ABI, tensor pointers, HPU allocations or device acquisition'}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

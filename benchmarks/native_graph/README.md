# Native graph experiment

`native.cpp` measures repeated one-node recipes versus one compiled multi-node
recipe. SCAL interposition is a closed research experiment, never a model backend.
Reverse paired order is `0,1,2,3,3,2,1,0`; 128 mutation gates check every output bit.

`owned_probe.cpp` exercises `csrc/native_graph/graph.h`: a Torch-free C ABI with
owned recipe clones, one graph memory pool, declared external memory, exact byte
range dependency validation, and bounded asynchronous flights with owned pinned
input/output storage and completion events. It destroys the producer's recipe,
frees original intermediate allocations, churns the allocator, inserts a larger
compiled graph, changes inputs, checks every FP32 output bit, and exercises
rejections and stale tickets. This is not a full-model or routing quality gate.

Build on the matching Gaudi host without acquiring a device:

```sh
python3 benchmarks/native_graph/build.py --output builds/native-graph
```

Run device programs through `tools/run_device_probe.py`, with fresh output cases
and matching committed `source-identity.json`. Do not bypass idle-device checks.

The Python `gaudi_kernels.graph` module provides explicit plan construction,
flight tickets, a bucket cache, and recipe/manifest save/load with runtime,
SDK-library, driver, and binary hash checks. Externals must be rebound with
retained owners when loading; saved physical addresses are never trusted.

V1 currently accepts explicit compiled recipes and AR/AG/RS commands, using
FP32/BF16/INT32/UINT8 collective payloads. It serializes all work in an owned
stream and owns the maximum recipe workspace. Multi-stream workspace reuse is
not supported. No precision policy or weight conversion is introduced.

`libgkg_capture.so` implements strict public recording, with explicit rejection
of unsupported HCCL collectives/P2P/groups. Successful standalone recording and
rejections have device evidence. Live Lazy framework draining is not qualified.

Still required before replacing the existing model backend: vLLM declarations
of live inputs, weights, KV and communicators; full-model allocator/workspace and
route-mutation gates; TP8 timing and quality; B×T/MTP integration. Merely using
this C ABI does not remove its per-recipe Synapse/HCCL CPU calls, and does not
constitute one mixed hardware submission.

Controlled binding/node experiments, TP8 collective evidence, real-model binding
ABBA, the status-30 root cause, and remaining model relocation work are recorded
in [NATIVE-GRAPH-20260930](../../docs/NATIVE-GRAPH-20260930.md).

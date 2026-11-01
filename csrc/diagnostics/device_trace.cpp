// Diagnostic-only Synapse trace exporter; no execution or launch interposition.
#include <synapse_api.h>
#include <cstdio>
#include <vector>
static synDeviceId device = 0;
struct Guard {};
#include "sdk_trace.inc"
extern "C" int trace_device(synDeviceId value) {
  if (profile_started) return -1;
  device = value;
  return 0;
}

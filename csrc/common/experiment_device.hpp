#pragma once
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

// Benchmarks must explicitly select a physical module. Library graph builders
// must take their device/graph from the framework and never acquire a device.
inline unsigned gaudi_experiment_module() {
    const char* value = std::getenv("GAUDI_KERNELS_MODULE_ID");
    char* end = nullptr;
    errno = 0;
    const unsigned long module = value ? std::strtoul(value, &end, 10) : ULONG_MAX;
    if (!value || !*value || *value == '-' || !end || *end || errno || module > 7) {
        std::fprintf(stderr, "GAUDI_KERNELS_MODULE_ID must explicitly select module 0..7\n");
        std::exit(2);
    }
    return static_cast<unsigned>(module);
}

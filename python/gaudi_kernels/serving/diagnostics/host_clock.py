"""Untimed identity for comparing process clocks on one Linux host."""
from pathlib import Path
import socket
import time


def clock_domain():
    return {'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'implementation': time.get_clock_info('perf_counter').implementation}

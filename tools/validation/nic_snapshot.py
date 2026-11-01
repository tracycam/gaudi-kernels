"""Read NIC masks and counters without opening devices or changing their state."""

import argparse
import json
from pathlib import Path
import re
import socket
import time


COUNTERS = (
    'pcs_link_toggles', 'pcs_link_restores', 'pcs_local_faults',
    'pcs_remote_faults', 'cw_uncorrect_accum', 'cw_corrected_accum',
    'requester_psn_out_of_range', 'responder_duplicate_psn',
    'responder_out_of_sequence_psn', 'tx_Octets', 'rx_Octets',
)


def rdma_to_hardware(ports_mask, rdma_ports):
    """hbl_main.c numbers only enabled hardware ports, starting at RDMA 1."""
    if ports_mask < 0:
        raise ValueError('ports_mask must be nonnegative')
    hardware = [p for p in range(ports_mask.bit_length()) if ports_mask & (1 << p)]
    if sorted(rdma_ports) != list(range(1, len(hardware) + 1)):
        raise ValueError('RDMA ports do not match ports_mask; refusing to guess hardware IDs')
    return dict(enumerate(hardware, start=1))


def read(path):
    try:
        return path.read_text().strip()
    except OSError as error:
        return {'error': str(error)}


def snapshot(sysfs_root=Path('/sys')):
    result = {'utc_ns': time.time_ns(), 'hostname': socket.gethostname(),
              'module_parameters': {}, 'devices': []}
    for module in ('habanalabs', 'habanalabs_cn', 'habanalabs_ib', 'habanalabs_en'):
        root = sysfs_root / 'module' / module
        result['module_parameters'][module] = {
            'loaded': root.exists(),
            'version': read(root / 'version'),
            'nic_ports_mask': read(root / 'parameters' / 'nic_ports_mask'),
        }
    for device in sorted((sysfs_root / 'class' / 'infiniband').glob('hbl_*')):
        ports_mask = read(device / 'ports_mask')
        ext_mask = read(device / 'ext_ports_mask')
        pci = next((part for part in reversed(device.resolve().parts)
                    if re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', part)), None)
        item = {'device': device.name, 'pci': pci,
                'ports_mask': ports_mask, 'ext_ports_mask': ext_mask, 'ports': []}
        try:
            rdma_paths = sorted((device / 'ports').iterdir(), key=lambda p: int(p.name))
            mapping = rdma_to_hardware(int(ports_mask, 16), [int(p.name) for p in rdma_paths])
            external = int(ext_mask, 16)
        except (ValueError, TypeError, OSError) as error:
            item['mapping_error'] = str(error)
            result['devices'].append(item)
            continue
        for rdma in rdma_paths:
            hardware = mapping[int(rdma.name)]
            item['ports'].append({
                'hardware_port': hardware, 'rdma_port': int(rdma.name),
                'external': bool(external & (1 << hardware)),
                'state': read(rdma / 'state'), 'physical_state': read(rdma / 'phys_state'),
                'counters': {field: read(rdma / 'hw_counters' / field) for field in COUNTERS},
            })
        result['devices'].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sysfs-root', type=Path, default=Path('/sys'))
    args = parser.parse_args()
    print(json.dumps(snapshot(args.sysfs_root), indent=2))


if __name__ == '__main__':
    main()

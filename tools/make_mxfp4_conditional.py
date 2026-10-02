"""Insert a once-per-task flag check before the unchanged historical MAC body."""
from pathlib import Path

def make(source: Path, destination: Path):
    assembly = source.read_text()
    marker = 'loop 0, S2, 1, <, .LBB0_6, SP1'
    assert assembly.count(marker) == 1
    nop = 'nop; nop; nop; nop\n'
    guard = ('set_indx I0, b11111, 0x0; nop; nop; nop\n' + nop * 4
             + 'nop; set_indx I0, b00010, S32; nop; nop\n' + nop * 4
             + 'nop; nop; nop; gen_addr AD0, 0x5, I0\n' + nop * 6
             + 'ld_g S26, AD0; nop; nop; nop\n' + nop * 8
             + 'nop; and.i32 S26, S26, 0xffff; nop; nop\n' + nop * 4
             + 'nop; cmp_eq.i32 SP1, S26, 0x0; nop; nop\n' + nop * 4)
    result = assembly.replace(marker, guard + marker).replace('st_tnsr 0x5,', 'st_tnsr 0x6,')
    assert result.split('.LBB0_3:')[1].split('.LBB0_6:')[0] == assembly.split('.LBB0_3:')[1].split('.LBB0_6:')[0]
    destination.write_text(result)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser();p.add_argument('source', type=Path);p.add_argument('output', type=Path);a=p.parse_args();make(a.source,a.output)

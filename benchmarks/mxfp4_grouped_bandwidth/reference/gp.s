.text
.globl main
main:
nop; mov.i32 S31, S0; nop; nop
ld_l mmio S28, 0xf0; nop; nop; nop
ld_l mmio S29, 0xf4; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; st_l mmio 0xcd8, S28
nop; nop; nop; st_l mmio 0xcdc, S29
nop; mov.i32 S27, 0x0; nop; nop
event vpu 0xff; 	nop; 	nop; 	nop
ld_l mmio S0, 0x538; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	or.i32  S0, S0, 0x1000000; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	st_l mmio 0x538, S0
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
mov.i32 S0, 2048; 	mov_irf_dim  0x0 S3, I0; 	nop; 	set_indx  I2, b11110, 0x0
set_indx  I3, b11111, 0x0; 	add.i32  b11111 I4, I1, I0; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	mov_irf_dim  0x0 S4, I4; 	nop; 	nop
nop; 	ash.i32  S2, S0, 0xffffffe1; 	nop; 	nop
nop; 	cmp_grt.i32  SP1, S0, 0x1f; 	nop; 	nop
nop; 	mul.i32  S1, S3, S0; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	shr.i32  S2, S2, 0x1b; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	add.i32  S2, S0, S2; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	ash.i32  S2, S2, 0xfffffffb; 	nop; 	nop
loop S3, S4, 1, <, .LBB0_7
nop; 	nop; 	nop; 	nop
.LBB0_2:
ld_l mmio S30, 0x150; nop; nop; nop
nop; mul.u32 upper32 S11, S32, 0xaaaaaaab; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; shr.u32 S11, S11, 1; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mul.i32 S12, S11, 3; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; sub.i32 S12, S32, S12; nop; nop
nop; mul.u32 upper32 S13, S11, S31; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mul.i32 S14, S13, S30; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; sub.i32 S14, S11, S14; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; cmp_geq.u32 SP2, S14, S30; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; add.i32 S13, S13, 1, SP2; nop; nop
nop; cmp_eq.i32 SP2, S30, 1; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mov.i32 S13, S11, SP2; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mul.i32 S14, S13, S30; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; sub.i32 S14, S11, S14; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
set_indx I0, b11111, 0x0; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; set_indx I0, b00001, S14; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; set_indx I0, b00010, S13; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; gen_addr AD0, 0x4, I0
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
ld_g S10, AD0; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mul.i32 S10, S10, 3; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; add.i32 S10, S10, S12; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; mul.i32 S1, S10, S0; nop; nop
nop; mul.i32 S3, S10, S2; nop; nop
nop; shl.i32 S15, S12, 0x6; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
mov.f32  V0, 0x0; 	shl.i32  S4, S32, 0x9; 	nop; 	nop
mov  b11111 I4, I2; 	shl.i32  S5, S32, 0x8; 	nop; 	nop
mov  b11111 I5, I2; 	nop; 	nop; 	nop
mov  b11111 I6, I3; nop; nop; nop
nop; set_indx I6, b00010, S13; nop; nop
set_indx  I4, b00001, S4; 	mov.f32  S4, S1; 	mov  V1, V0; 	nop
set_indx  I5, b00001, S5; 	mov  b11111 I7, I4; 	mov  V2, V0; 	nop
mov  V4, V0; 	nop; 	mov  V6, V0; 	nop
nop; 	nop; 	nop; 	nop
mov  V7, V1; 	nop; 	mov  V3, V1; 	nop
nop; 	nop; 	mov  V5, V1; 	nop
loop 0, S2, 1, <, .LBB0_6, SP1
nop; 	nop; 	nop; 	nop
.LBB0_3:
nop; add.i32 S7, S33, S15; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; shl.i32 S7, S7, 0x5; nop; nop
mov.f32 V14, 0x0; set_indx I5, b00010, S4; mov.f32 V15, 0x0; nop
mov.f32 V12, 0x0; set_indx I7, b00001, 0x0; mov.f32 V13, 0x0; set_indx I5, b00001, 0x0
mov.f32 V10, 0x0; set_indx I7, b00010, S33; mov.f32 V11, 0x0; nop
mov.f32 V8, 0x0; add.i32 b00010 I7, S3, I7; mov.f32 V9, 0x0; set_indx I6, b00001, S7
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
ld_tnsr V39, 0x2, I6; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; nop; nop
mov.i32 V34, 0x81808180; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V38, 0x1, I7; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V39, V39, 0xffffffff; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0x83828382; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V34, 0x85848584; nop; nop; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; nop; nop
nop; nop; nop; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; nop; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V35, 0x87868786; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V34, 0x89888988; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V35, 0x8b8a8b8a; nop; mac.bf16 acc_fp32 D14, V30, V36; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V36; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V34, 0x8d8c8d8c; nop; mac.bf16 acc_fp32 D14, V18, V37; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V37; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0x8f8e8f8e; nop; mac.bf16 acc_fp32 D14, V24, V36; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V36; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V34, 0x91909190; nop; mac.bf16 acc_fp32 D14, V30, V37; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V37; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V35, 0x93929392; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V34, 0x95949594; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V35, 0x97969796; nop; mac.bf16 acc_fp32 D14, V30, V36; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V36; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V34, 0x99989998; nop; mac.bf16 acc_fp32 D14, V18, V37; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V37; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0x9b9a9b9a; nop; mac.bf16 acc_fp32 D14, V24, V36; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V36; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V34, 0x9d9c9d9c; nop; mac.bf16 acc_fp32 D14, V30, V37; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V37; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V35, 0x9f9e9f9e; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V34, 0xa1a0a1a0; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V35, 0xa3a2a3a2; nop; mac.bf16 acc_fp32 D14, V30, V36; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V36; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V34, 0xa5a4a5a4; nop; mac.bf16 acc_fp32 D14, V18, V37; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V37; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0xa7a6a7a6; nop; mac.bf16 acc_fp32 D14, V24, V36; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V36; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V34, 0xa9a8a9a8; nop; mac.bf16 acc_fp32 D14, V30, V37; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V37; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V35, 0xabaaabaa; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V34, 0xadacadac; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V35, 0xafaeafae; nop; mac.bf16 acc_fp32 D14, V30, V36; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V36; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V34, 0xb1b0b1b0; nop; mac.bf16 acc_fp32 D14, V18, V37; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V37; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0xb3b2b3b2; nop; mac.bf16 acc_fp32 D14, V24, V36; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V36; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V34, 0xb5b4b5b4; nop; mac.bf16 acc_fp32 D14, V30, V37; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V37; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V35, 0xb7b6b7b6; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V34, 0xb9b8b9b8; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
mov.i32 V35, 0xbbbabbba; nop; mac.bf16 acc_fp32 D14, V30, V36; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V36; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
mov.i32 V34, 0xbdbcbdbc; nop; mac.bf16 acc_fp32 D14, V18, V37; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V37; nop
nop; add.i32 b00001 I7, 0x100, I7; mac.bf16 acc_fp32 D10, V20, V37; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V37; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
mov.i32 V35, 0xbfbebfbe; nop; mac.bf16 acc_fp32 D14, V24, V36; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V36; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, V36; nop
nop; nop; shuffle.u8 V36, V39, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7; nop; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
nop; nop; mac.bf16 acc_fp32 D14, V30, V37; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, V37; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, V37; nop
nop; nop; convert.u8 all_lanes target_type=uint16 rhne D16, V38; nop
nop; nop; shuffle.u8 V37, V39, V35; nop
nop; nop; mac.bf16 acc_fp32 D14, V18, V36; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, V36; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, V36; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, V36; nop
nop; nop; convert.u8 all_lanes target_type=uint16 rhne D22, V34; nop
nop; nop; shl.i16 V16, V16, 0x7; nop
nop; nop; mac.bf16 acc_fp32 D14, V24, V37; nop
nop; nop; mac.bf16 acc_fp32 D12, V25, V37; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, V37; nop
nop; nop; mac.bf16 acc_fp32 D8, V27, V37; nop
nop; add.i32 S4, S4, 0x20; shl.i16 V17, V17, 0x7; nop
nop; nop; shl.i16 V22, V22, 0x7; nop
nop; nop; shl.i16 V23, V23, 0x7; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D28, V16; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D30, V17; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D32, V22; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D34, V23; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; mac.f32 V0, V14, V28; nop
nop; nop; mac.f32 V1, V15, V29; nop
nop; nop; mac.f32 V2, V12, V30; nop
nop; nop; mac.f32 V3, V13, V31; nop
nop; nop; mac.f32 V6, V10, V32; nop
nop; nop; mac.f32 V7, V11, V33; nop
nop; nop; mac.f32 V4, V8, V34; nop
nop; nop; mac.f32 V5, V9, V35; nop
.LBB0_6:
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; add.i32 S1, S1, S0; nop; nop
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V0
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V1
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V2
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V3
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V6
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V7
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V4
nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x5, I4, V5
.LBB0_7:
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	halt; 	halt; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop
nop; 	nop; 	nop; 	nop

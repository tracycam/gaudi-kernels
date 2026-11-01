.text
.globl main
main:
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
ld_l mmio S0, 0xb0; 	mov_irf_dim  0x0 S3, I0; 	nop; 	set_indx  I2, b11110, 0x0
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
set_indx I0, b11111, 0x0; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; set_indx I0, b00001, S32; nop; nop
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
nop; mul.i32 S1, S10, S0; nop; nop
nop; mul.i32 S3, S10, S2; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; cmp_geq.i32 SP2, S10, 0; nop; nop
nop; cmp_grt.i32 SP1, S0, 31; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; and.b SP1, SP1, SP2; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
mov.f32  V0, 0x0; 	shl.i32  S4, S32, 0x9; 	nop; 	nop
mov  b11111 I4, I2; 	shl.i32  S5, S32, 0x8; 	nop; 	nop
mov  b11111 I5, I2; 	nop; 	nop; 	nop
mov  b11111 I6, I3; nop; nop; nop
nop; set_indx I6, b00010, S32; nop; nop
set_indx  I4, b00001, S4; 	mov.f32  S4, S1; 	mov  V1, V0; 	nop
set_indx  I5, b00001, S5; 	mov  b11111 I7, I4; 	mov  V2, V0; 	nop
mov  V4, V0; 	nop; 	mov  V6, V0; 	nop
nop; 	nop; 	nop; 	nop
mov  V7, V1; 	nop; 	mov  V3, V1; 	nop
nop; 	nop; 	mov  V5, V1; 	nop
loop 0, S2, 1, <, .LBB0_6, SP1
nop; 	nop; 	nop; 	nop
.LBB0_3:
nop; shl.i32 S7, S33, 0x5; nop; nop
mov.f32 V14, 0x0; set_indx I5, b00010, S4; mov.f32 V15, 0x0; nop
mov.f32 V12, 0x0; set_indx I7, b00001, 0x0; mov.f32 V13, 0x0; set_indx I5, b00001, 0x0
mov.f32 V10, 0x0; set_indx I7, b00010, S33; mov.f32 V11, 0x0; nop
mov.f32 V8, 0x0; add.i32 b00010 I7, S3, I7; mov.f32 V9, 0x0; set_indx I6, b00001, S7
nop; nop; nop; gen_addr dt=int8 AD0, 0x2, I6
nop; add.i32 b00001 I6, 0x1, I6; nop; nop
nop; nop; nop; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; nop; nop
ld_g S16, AD0; add.i32 b00001 I6, 0x1, I6; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7; add.i32 b00001 I7, 0x100, I7; nop; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V36, 0x1, I7; nop; nop; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S17, AD1; add.i32 b00001 I6, 0x1, I6; nop; nop
nop; nop; nop; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; nop; nop
nop; nop; nop; nop
nop; nop; convert.u8 all_lanes target_type=uint16 rhne D34, V34; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S18, AD2; add.i32 b00001 I6, 0x1, I6; nop; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; nop; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; nop; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; nop; nop
nop; nop; convert.u8 all_lanes target_type=uint16 rhne D36, V36; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S19, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S16; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S16; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S16; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S16; nop
nop; nop; shl.i16 V34, V34, 0x7; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S20, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S17; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S17; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S17; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S17; nop
nop; nop; shl.i16 V35, V35, 0x7; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S16, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S18; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S18; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S18; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S18; nop
nop; nop; shl.i16 V36, V36, 0x7; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S17, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S19; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S19; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S19; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S19; nop
nop; nop; shl.i16 V37, V37, 0x7; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S18, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S20; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S20; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S20; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S20; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S19, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S16; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S16; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S16; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S16; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S20, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S17; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S17; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S17; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S17; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S16, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S18; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S18; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S18; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S18; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S17, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S19; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S19; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S19; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S19; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S18, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S20; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S20; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S20; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S20; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S19, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S16; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S16; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S16; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S16; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S20, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S17; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S17; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S17; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S17; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S16, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S18; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S18; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S18; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S18; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S17, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S19; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S19; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S19; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S19; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S18, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S20; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S20; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S20; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S20; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S19, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S16; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S16; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S16; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S16; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S20, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S17; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S17; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S17; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S17; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S16, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S18; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S18; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S18; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S18; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S17, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S19; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S19; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S19; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S19; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S18, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S20; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S20; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S20; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S20; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S19, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S16; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S16; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S16; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S16; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S20, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S17; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S17; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S17; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S17; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S16, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S18; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S18; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S18; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S18; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S17, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S19; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S19; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S19; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S19; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S18, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S20; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S20; gen_addr dt=int8 AD2, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V20, S20; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S20; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S19, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S16; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S16; gen_addr dt=int8 AD0, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V26, S16; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S16; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V28, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
ld_g S20, AD2; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V30, S17; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S17; gen_addr dt=int8 AD1, 0x2, I6
nop; nop; mac.bf16 acc_fp32 D10, V32, S17; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S17; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V16, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D28, V28; nop
ld_g S16, AD0; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V18, S18; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S18; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, S18; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S18; nop
nop; nop; nop; nop
ld_tnsr UNPCK_8_TO_16 unpack V22, 0x0, I5; add.i32 b00010 I5, 0x1, I5; convert.u8 all_lanes target_type=uint16 rhne D16, V16; nop
ld_g S17, AD1; add.i32 b00001 I6, 0x1, I6; mac.bf16 acc_fp32 D14, V24, S19; nop
lookup_2c BV16 lut_ptr D30, V28, S27; nop; mac.bf16 acc_fp32 D12, V25, S19; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, S19; nop
lookup_2c BV16 lut_ptr D32, V29, S27; nop; mac.bf16 acc_fp32 D8, V27, S19; nop
nop; nop; nop; nop
nop; nop; convert.u8 all_lanes target_type=uint16 rhne D22, V22; nop
nop; nop; mac.bf16 acc_fp32 D14, V30, S20; nop
lookup_2c BV16 lut_ptr D18, V16, S27; nop; mac.bf16 acc_fp32 D12, V31, S20; nop
nop; nop; mac.bf16 acc_fp32 D10, V32, S20; nop
lookup_2c BV16 lut_ptr D20, V17, S27; nop; mac.bf16 acc_fp32 D8, V33, S20; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D28, V34; nop
nop; nop; nop; nop
nop; nop; mac.bf16 acc_fp32 D14, V18, S16; nop
lookup_2c BV16 lut_ptr D24, V22, S27; nop; mac.bf16 acc_fp32 D12, V19, S16; nop
nop; nop; mac.bf16 acc_fp32 D10, V20, S16; nop
lookup_2c BV16 lut_ptr D26, V23, S27; nop; mac.bf16 acc_fp32 D8, V21, S16; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D30, V35; nop
nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D32, V36; nop
nop; nop; mac.bf16 acc_fp32 D14, V24, S17; nop
nop; nop; mac.bf16 acc_fp32 D12, V25, S17; nop
nop; nop; mac.bf16 acc_fp32 D10, V26, S17; nop
nop; nop; mac.bf16 acc_fp32 D8, V27, S17; nop
nop; add.i32 S4, S4, 0x20; convert.bf16 all_lanes target_type=fp32 rhne D34, V37; nop
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

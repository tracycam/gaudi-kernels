__local volatile float64 combined[8];
// ELF descriptors only: executable .text is the scheduled assembly wrapper.
void main(tensor packed,tensor scales,tensor activation,tensor lut,tensor ids,
          tensor routing,tensor output,int nblocks,unsigned reciprocal,int experts,unsigned route_reciprocal){
 set_lut_256(lut);int5 p={0,0,0,0,0};
 float64 a=v_f32_ld_tnsr_b(p,packed),b=v_f32_ld_tnsr_b(p,scales),c=v_f32_ld_tnsr_b(p,activation);
 a+=v_f32_ld_tnsr_b(p,ids)+v_f32_ld_tnsr_b(p,routing)+b+c+(float)(nblocks+reciprocal+experts+route_reciprocal);
 for(int i=0;i<8;++i)combined[i]=a;
 v_f32_st_tnsr(p,output,combined[7]);
}

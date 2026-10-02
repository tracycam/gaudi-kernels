// Logical traffic probe, not a physical bus counter. Audit looped loads/stores.
// One independent 256-byte lane vector per task; rows is a multiple of eight.
#ifndef PORT_WRITE_UNROLL
#define PORT_WRITE_UNROLL 8
#endif
#if PORT_WRITE_UNROLL != 8 && PORT_WRITE_UNROLL != 32
#error unsupported write unroll
#endif
void main(tensor src, tensor dst, int rows, int repeats) {
    int5 begin=get_index_space_offset(), end=begin+get_index_space_size();
    for(int task=begin[0];task<end[0];++task) {
        uint64 a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        for(int r=0;r<repeats;++r) {
            for(int j=0;j<rows;j+=(PORT_MODE == 1 ? PORT_WRITE_UNROLL : 8)) {
#ifdef PORT_FLAT
                // Identical byte ownership to the contiguous 3-D descriptor.
                // Collapse the task coordinate before the tensor address unit.
                int5 p={0,task*rows+j,0,0,0};
#define ROWDIM 1
#elif defined(PORT_INTERLEAVE)
                int5 p={0,task,j,0,0};
#define ROWDIM 2
#else
                int5 p={0,j,task,0,0};
#define ROWDIM 1
#endif
#if PORT_MODE == 0
#define STEP(A) A+=v_u32_ld_tnsr_b(p,src); p[ROWDIM]+=1;
                STEP(a0) STEP(a1) STEP(a2) STEP(a3)
                STEP(a4) STEP(a5) STEP(a6) STEP(a7)
#elif PORT_MODE == 1
                uint64 v=(unsigned)(r+task+1);
#define STEP(A) v_u32_st_tnsr(p,dst,v); p[ROWDIM]+=1;
                STEP(a0) STEP(a1) STEP(a2) STEP(a3)
                STEP(a4) STEP(a5) STEP(a6) STEP(a7)
#if PORT_WRITE_UNROLL == 32
                STEP(a0) STEP(a1) STEP(a2) STEP(a3)
                STEP(a4) STEP(a5) STEP(a6) STEP(a7)
                STEP(a0) STEP(a1) STEP(a2) STEP(a3)
                STEP(a4) STEP(a5) STEP(a6) STEP(a7)
                STEP(a0) STEP(a1) STEP(a2) STEP(a3)
                STEP(a4) STEP(a5) STEP(a6) STEP(a7)
#endif
#else
                int5 q=p;
#define LOAD(V) uint64 V=v_u32_ld_tnsr_b(p,src); p[ROWDIM]+=1;
                LOAD(v0) LOAD(v1) LOAD(v2) LOAD(v3)
                LOAD(v4) LOAD(v5) LOAD(v6) LOAD(v7)
#define STORE(V) v_u32_st_tnsr(q,dst,V+(unsigned)r); q[ROWDIM]+=1;
                STORE(v0) STORE(v1) STORE(v2) STORE(v3)
                STORE(v4) STORE(v5) STORE(v6) STORE(v7)
#undef LOAD
#undef STORE
#endif
#undef STEP
#undef ROWDIM
            }
        }
#if PORT_MODE == 0
        int5 q={0,task,0,0,0};
        v_u32_st_tnsr(q,dst,(a0+a1)+(a2+a3)+(a4+a5)+(a6+a7));
#endif
    }
}

// One index-space task. Every output cell is written, including failure paths.
void main(tensor counts,tensor row_status,tensor prefix,tensor tile_expert,
          tensor tile_base,tensor valid_rows,tensor status,int routes,int rows,int capacity,int min_rows,int max_rows){
 int experts=get_dim_size(counts,0),tokens=get_dim_size(row_status,0);
 int flags=0,total=0,tiles=0;
 for(int t=0;t<tokens;++t){int5 c={t,0,0,0,0};int f=s_i32_ld_g(gen_addr(c,row_status));flags|=f;if(f<0||f>3)flags|=8;}
 for(int e=0;e<experts;++e){
  int5 c={e,0,0,0,0};int n=s_i32_ld_g(gen_addr(c,counts));
  if(n<0||n>tokens)flags|=8;else{total+=n;if(n>=min_rows&&n<=max_rows)++tiles;}
 }
 if(total!=tokens*routes)flags|=8;
 if(tiles>capacity)flags|=4;
 int cursor=0;
 for(int e=0;e<experts;++e){
  int5 c={e,0,0,0,0};s_i32_st_g(gen_addr(c,prefix),cursor);
  int n=s_i32_ld_g(gen_addr(c,counts));
  if(flags==0&&n>=min_rows&&n<=max_rows){
   int5 d={cursor,0,0,0,0};s_i32_st_g(gen_addr(d,tile_expert),e);
   s_i32_st_g(gen_addr(d,tile_base),0);
   s_i32_st_g(gen_addr(d,valid_rows),n);++cursor;
  }
 }
 int5 last={experts,0,0,0,0};s_i32_st_g(gen_addr(last,prefix),cursor);
 for(int tile=cursor;tile<capacity;++tile){int5 c={tile,0,0,0,0};
  s_i32_st_g(gen_addr(c,tile_expert),-1);s_i32_st_g(gen_addr(c,tile_base),0);s_i32_st_g(gen_addr(c,valid_rows),0);
 }
 int5 z={0,0,0,0,0};s_i32_st_g(gen_addr(z,status),flags);
}

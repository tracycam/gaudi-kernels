// One index-space task. Every output cell is written, including failure paths.
void main(tensor counts,tensor row_status,tensor prefix,tensor tile_expert,
          tensor tile_base,tensor valid_rows,tensor status,int routes,int rows,int capacity,int min_rows,int max_rows,int overflow_to_tpc){
 int experts=get_dim_size(counts,0),tokens=get_dim_size(row_status,0);
 int flags=0,total=0,tiles=0;
 for(int t=0;t<tokens;++t){int5 c={t,0,0,0,0};int f=s_i32_ld_g(gen_addr(c,row_status));flags|=f;if(f<0||f>3)flags|=8;}
 for(int e=0;e<experts;++e){
  int5 c={e,0,0,0,0};int n=s_i32_ld_g(gen_addr(c,counts));
  if(n<0||n>tokens)flags|=8;else{total+=n;if(n>=min_rows&&n<=max_rows)++tiles;}
 }
 if(total!=tokens*routes)flags|=8;
 if(tiles>capacity && !overflow_to_tpc)flags|=4;
 int cutoff=min_rows,ties=capacity;
#ifdef GK_EXPERT_LARGEST_M
 // Capped selection prioritizes actual M within this row band. Binary search
 // needs O(E log(row range)) scalar loads and no private sorting workspace.
 // Retain expert order for prefix/inverse; equal M breaks ties by expert ID.
 if(flags==0&&tiles>capacity&&overflow_to_tpc){
  int lo=min_rows,hi=max_rows;
  while(lo<hi){
   int mid=(lo+hi+1)/2,above=0;
   for(int e=0;e<experts;++e){int5 c={e,0,0,0,0};int n=s_i32_ld_g(gen_addr(c,counts));if(n>=mid&&n<=max_rows)++above;}
   if(above>=capacity)lo=mid;else hi=mid-1;
  }
  cutoff=lo;
  int above=0;
  for(int e=0;e<experts;++e){int5 c={e,0,0,0,0};int n=s_i32_ld_g(gen_addr(c,counts));if(n>cutoff&&n<=max_rows)++above;}
  ties=capacity-above;
 }
#endif
 int cursor=0;
 for(int e=0;e<experts;++e){
  int5 c={e,0,0,0,0};s_i32_st_g(gen_addr(c,prefix),cursor);
  int n=s_i32_ld_g(gen_addr(c,counts));
  int chosen=n>=min_rows&&n<=max_rows;
#ifdef GK_EXPERT_LARGEST_M
  chosen=chosen&&(n>cutoff||(n==cutoff&&ties>0));
  if(chosen&&n==cutoff)--ties;
#endif
  if(flags==0&&chosen&&cursor<capacity){
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

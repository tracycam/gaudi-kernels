#include <cmath>
#include <cstdint>
#include <cstring>
extern "C" void reduce_cpu(const float*partial,const float*as,const float*ws,const float*bias,float*out,uint16_t*bf,int groups,int rows,int cols,int chains){
 for(int m=0;m<rows;++m)for(int n=0;n<cols;++n){float acc[8]={};
  if(chains==0){float sum=0,correction=0;
   for(int g=0;g<groups;++g){float factor=as[g*rows+m]*ws[(n/128)*groups+g],v=partial[(g*rows+m)*cols+n],term=v*factor,error=std::fma(v,factor,-term),updated=sum+term;
    float residual=std::abs(sum)>=std::abs(term)?(sum-updated)+term:(term-updated)+sum;
    correction=correction+(residual+error);sum=updated;
   }acc[0]=sum+correction;
  }else{
  for(int g=0;g<groups;++g){float factor=as[g*rows+m]*ws[(n/128)*groups+g];acc[g%chains]=std::fma(partial[(g*rows+m)*cols+n],factor,acc[g%chains]);}
  for(int step=1;step<chains;step*=2)for(int c=0;c<chains;c+=step*2)acc[c]=acc[c]+acc[c+step];
  }
  float result=acc[0]+bias[n];out[m*cols+n]=result;uint32_t bits;std::memcpy(&bits,&result,4);bf[m*cols+n]=(bits+0x7fff+((bits>>16)&1))>>16;
 }
}

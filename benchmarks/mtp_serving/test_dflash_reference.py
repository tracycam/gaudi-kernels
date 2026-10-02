import unittest
import torch
from dflash_reference import attention,rope,partition


class Contract(unittest.TestCase):
    def test_tp_projection_axes_preserve_exact_checkpoint_values(self):
        torch.manual_seed(20);x=torch.randn(3,16);weight=torch.randn(32,16)
        for key in ('fc.weight','layers.0.self_attn.q_proj.weight','layers.0.mlp.gate_proj.weight'):
            outputs=[x@weight[partition(key,weight.shape,r,8)].T for r in range(8)]
            torch.testing.assert_close(torch.cat(outputs,-1),x@weight.T)
        for key in ('layers.0.self_attn.o_proj.weight','layers.0.mlp.down_proj.weight'):
            outputs=[x[:,r*2:(r+1)*2]@weight[partition(key,weight.shape,r,8)].T for r in range(8)]
            torch.testing.assert_close(sum(outputs),x@weight.T,rtol=2e-6,atol=3e-6)
        gamma=torch.randn(16)
        for r in range(8):self.assertTrue(torch.equal(gamma[partition('layers.0.self_attn.q_norm.weight',gamma.shape,r,8)],gamma))
        sink=torch.randn(32)
        parts=[sink[partition('layers.0.self_attn.attention_sink_bias',sink.shape,r,8)]for r in range(8)]
        self.assertTrue(torch.equal(torch.cat(parts),sink))

    def test_symmetric_window_sink_and_gqa(self):
        torch.manual_seed(19)
        q=torch.randn(2,3,4,8);k=torch.randn(2,5,2,8);v=torch.randn_like(k)
        qp=torch.tensor([[1023,1024,1025],[7,8,9]])
        kp=torch.tensor([[0,1,1023,1024,1025],[0,7,8,9,10]])
        valid=torch.tensor([[1,1,1,1,1],[0,1,0,1,1]],dtype=torch.bool)
        sink=torch.tensor([-1.,0.,1.,3.])
        actual=attention(q,k,v,qp,kp,valid,sink,1024).reshape(2,3,4,8)
        expected=torch.empty_like(actual)
        for b in range(2):
            for t in range(3):
                selected=[j for j in range(5)if valid[b,j]and abs(int(qp[b,t]-kp[b,j]))<1024]
                for h in range(4):
                    logits=torch.stack([torch.dot(q[b,t,h],k[b,j,h//2])*(8**-.5)for j in selected]+[sink[h]])
                    probs=logits.softmax(0)
                    expected[b,t,h]=sum(probs[n]*v[b,j,h//2]for n,j in enumerate(selected))
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
        # A sink-only row contributes exactly zero, with no NaN propagation.
        empty=attention(q,k,v,qp,kp,torch.zeros_like(valid),sink,1024)
        self.assertTrue(torch.equal(empty,torch.zeros_like(empty)))
        # Noncausal: changing a future draft V changes an earlier query.
        vv=v.clone();vv[0,-1]+=10
        changed=attention(q,k,vv,qp,kp,valid,sink,1024)
        self.assertFalse(torch.equal(changed[0,0],actual.reshape(2,3,32)[0,0]))

    def test_partial_rope_fp32_oracle(self):
        x=torch.arange(16.).reshape(1,1,2,8).to(torch.bfloat16);p=torch.tensor([[1031]])
        actual=rope(x,p,4,10000.)
        expected=x.float().clone()
        for h in range(2):
            for j in range(2):
                a=torch.tensor(1031*(10000**(-j/2)))
                u,w=x[0,0,h,j].float(),x[0,0,h,j+2].float()
                expected[0,0,h,j]=u*a.cos()-w*a.sin()
                expected[0,0,h,j+2]=u*a.sin()+w*a.cos()
        self.assertTrue(torch.equal(actual,expected.to(x.dtype)))
        self.assertTrue(torch.equal(actual[...,4:],x[...,4:]))


if __name__=='__main__':unittest.main()

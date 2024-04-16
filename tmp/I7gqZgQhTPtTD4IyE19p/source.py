import torch

def trilinear_composition(h_s, h_t, x, einsum=True):
    
    if einsum:
        return torch.einsum("lbe,lbe,ve->blv", h_s, h_t, x)

    else:
        b = h_s.shape[1]

        # h_t ⊙ x
        # l,b,e ⊙ v,1,1,e  (so b,e dimensions are broadcasted)
        bc = h_t * x.reshape(x.shape[0], 1, 1, x.shape[1])

        # aT @ (bc) = h_sT @ (h_t ⊙ x)
        # ---------
        # l,b,e @ v,l,e,b
        # b,e @ e,b will be matrix multiplied. v,l are untouched.
        # result is a v,l,b,b matrix
        vlbb = torch.matmul(h_s, bc.transpose(-1, -2))

        # As dot products should not be applied across batches, the diagonal can be selected
        # v,l,b
        vlb = vlbb[..., torch.arange(0, b), torch.arange(0, b)]
        # b,l,v
        return vlb.transpose(0, 2)
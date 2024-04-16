import torch

def gradient(u, f, method='finitediff', eps=1e-4):
    
    if method == 'autodiff':
        with torch.enable_grad():
            u = u.requires_grad_(True)
            v = f(u)
            grad = torch.autograd.grad(v, u, 
                                       grad_outputs=torch.ones_like(v), create_graph=True)[0]
    elif method == 'finitediff':
        assert(u.shape[-1] == 2 and "Finitediff only supports 2D vector fields")
        eps_x = torch.tensor([eps, 0.0], device=u.device)
        eps_y = torch.tensor([0.0, eps], device=u.device)

        grad = torch.cat([f(u + eps_x) - f(u - eps_x),
                          f(u + eps_y) - f(u - eps_y)], dim=-1)
        grad = grad / (eps*2.0)
    else:
        raise NotImplementedError

    return grad
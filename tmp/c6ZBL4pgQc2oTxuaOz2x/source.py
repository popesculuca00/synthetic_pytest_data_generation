import torch

def compas_robustness_loss(x, aggregates, concepts, relevances):
    
    batch_size = x.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(x.device)
    J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
                               grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances

    return robustness_loss.norm(p='fro')
import torch

def one_hot(indices, depth):
    

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies
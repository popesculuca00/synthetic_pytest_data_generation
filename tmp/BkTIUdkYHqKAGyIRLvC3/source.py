import torch

def prepare_bimodal_siamese_tensors(embedding_list_0, embedding_list_1, siamese_target):
    
    embedding_size = embedding_list_0[0].shape[1]
    siamese_pair_0 = torch.stack(embedding_list_0).repeat_interleave(2, 0).view(-1, embedding_size)
    siamese_pair_1 = torch.stack(embedding_list_1).repeat(2, 1, 1).view(-1, embedding_size)
    siamese_target = siamese_target.repeat(4)
    return siamese_pair_0, siamese_pair_1, siamese_target
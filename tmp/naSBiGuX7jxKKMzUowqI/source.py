import torch

def construct_edge_feature_gather(feature, knn_inds):
    
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_expand = feature.unsqueeze(2).expand(batch_size, channels, num_nodes, num_nodes)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = torch.gather(feature_expand, 3, knn_inds_expand)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature
def activation_channels_means(activation):
    
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_means = view_2d.mean(dim=1)  # (batch*channels) x 1
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_means_mat.mean(dim=0).cpu()
import torch

def match_corr(embed_ref, embed_srch):
    

    _, _, k1, k2 = embed_ref.shape
    b, c, h, w = embed_srch.shape

    if k1 == 1 and k2 == 1:
        pad_img = (0, 0)
    else:
        pad_img = (0, 1)
    match_map = torch.nn.functional.conv2d(embed_srch.contiguous().view(1, b * c, h, w), embed_ref, groups=b, padding=pad_img)

    match_map = match_map.permute(1, 0, 2, 3)

    return match_map
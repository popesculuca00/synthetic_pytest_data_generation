import torch

def reduceByDepth(rgb_images, depth_images, max_depth=None):
    

    label_image = depth_images.argmin(-3)
    new_shape = label_image.shape

    num_batch = new_shape[0]
    num_rows, num_cols = new_shape[-2:]
    b, r, c = torch.meshgrid(
        torch.arange(num_batch),
        torch.arange(num_rows),
        torch.arange(num_cols)
    )
    i_min = label_image.contiguous().view(-1)
    b = b.contiguous().view(-1)
    r = r.contiguous().view(-1)
    c = c.contiguous().view(-1)

    depth_image = depth_images[b, i_min, r, c].view(*new_shape)
    rgb_image = rgb_images[b, i_min, r, c, :].view(*new_shape, 3)

    if max_depth is not None:
        label_image += 1
        is_background = depth_image == max_depth
        label_image[is_background] = 0

    return rgb_image, depth_image, label_image
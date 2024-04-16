def pixel_shuffle_inv(tensor, scale_factor):
    
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor
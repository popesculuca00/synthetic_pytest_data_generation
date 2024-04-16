def dequantize(x, scale_factor, n_bits=8):
    
    min_level = -(1 << (n_bits - 1))
    max_level = (1 << (n_bits - 1)) - 1
    integer_range = 1 << (n_bits - 1)

    # check for overflow
    if x.min() < min_level or x.max() > max_level:
        raise OverflowError()

    x = x / integer_range
    x = x * scale_factor
    return x
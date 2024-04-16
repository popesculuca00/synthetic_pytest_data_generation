def get_input_dim(dim1, dim2):
    
    dim1 = [dim1] if isinstance(dim1, int) else dim1
    dim2 = [dim2] if isinstance(dim2, int) else dim2
    if len(dim1)==1 and len(dim2)==1:
        out_dim = [dim1[0] + dim2[0]]
    elif len(dim1)==3 and len(dim2)==1:
        out_dim = [dim1[0]+dim2[0], *dim1[1:]]
    elif len(dim1)==1 and len(dim2)==3:
        out_dim = [dim2[0]+dim1[0], *dim2[1:]]
    elif len(dim1)==3 and len(dim2)==3:
        assert (dim1[1] == dim2[1]) and (dim1[2] == dim2[2]), (
            "If both dim1 and dim2 are arrays, must have same shape. dim1: {}. dim2: {}.".format(dim1, dim2)
        )
        out_dim = [dim1[0]+dim2[0], *dim1[1:]]
    else:
        raise AssertionError("dim1 and dim2 must have length one or three. Given: {} and {}.".format(dim1, dim2))
    return tuple(out_dim)
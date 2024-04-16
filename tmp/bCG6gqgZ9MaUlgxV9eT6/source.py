def _get_pad_left_right(small, large):
    
    assert small < large, "Can only pad when new size larger than old size"

    padsize = large - small
    if padsize % 2 != 0:
        leftpad = (padsize - 1)/2
    else:
        leftpad = padsize/2
    rightpad = padsize-leftpad

    return int(leftpad), int(rightpad)
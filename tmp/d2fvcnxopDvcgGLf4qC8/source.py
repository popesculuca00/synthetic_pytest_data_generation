def tw_kern(x, m, h):
    
    z = (x - m) / h
    if z < -3 or z > 3:
        return 0
    else:
        return 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h
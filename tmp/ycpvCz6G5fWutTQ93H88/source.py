def _recall_bias(df, frac):
    
    df.sort()

    tiny = 1e-7
    if frac <= 0:
        return df[-1] + tiny
    if frac >= 1:
        return df[0]

    ind = int((1 - frac) * df.size)
    return df[ind]
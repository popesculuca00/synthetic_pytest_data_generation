def exact_CI(k, n, conf=0.683):
    

    from scipy.stats import beta
    k = float(k)
    n = float(n)
    p = (k/n) if n > 0 else 0

    alpha = (1 - conf)
    up = 1 if k == n else 1 - beta.ppf(alpha/2, n-k, k+1)
    down = 0 if k == 0 else 1 - beta.ppf(1-alpha/2, n-k+1, k)

    result = (p, p-down, up-p)
    return result
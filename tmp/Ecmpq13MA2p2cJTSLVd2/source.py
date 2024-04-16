def incremental_search(f, a, b, dx):
    
    x1 = a
    x2 = x1 + dx

    f1 = f(x1)
    f2 = f(x2)

    while (f1*f2) > 0:
        if x1 >= b:
            return None, None

        x1 = x2
        x2 = x1 + dx

        f1 = f2
        f2 = f(x2)

    return x1, x2
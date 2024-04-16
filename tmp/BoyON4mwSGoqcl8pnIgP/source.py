def _airtovac(w):
    
    if w < 2000.0:
        return w;
    vac = w
    for iter in range(2):
        sigma2 = (1.0e4/vac)*(1.0e4/vac)
        fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
        vac = w*fact
    return vac
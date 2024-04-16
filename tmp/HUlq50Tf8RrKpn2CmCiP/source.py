def delta_value(decimals=1, unit='$'):
    
    return (lambda a, b: '{:+,.{prec}f} {unit}'.format(1.0 * (b - a), unit=unit, prec=decimals))
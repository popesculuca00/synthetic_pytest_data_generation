def boost_nph(group, nph):
    
    n = len(group[group['binding_score'] <= 1.0])
    return round(nph * ((n >= 1) * 0.4 + (n >= 2) * 0.3 + (n >= 3) * 0.2 + (n >= 4) * 0.1), 2)
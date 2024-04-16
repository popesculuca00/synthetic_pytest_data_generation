def find_high_frequency_categories(s, min_frequency=0.02, n_max=None):
    
    assert 0.0 < min_frequency < 1.0
    s = s.value_counts(normalize=True).pipe(lambda s: s[s > min_frequency])

    if n_max is None:
        return list(s.index)

    if len(s) <= n_max:
        return s

    return list(s.sort_values(ascending=False).iloc[:n_max].index)
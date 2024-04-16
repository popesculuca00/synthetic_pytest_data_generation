def map_onto_scale(p1, p2, s1, s2, v):
    
    assert p1 <= p2, (p1, p2)
    assert s1 <= s2, (s1, s2)
    if p1 == p2:
        assert s1 == s2, (p1, p2, s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r
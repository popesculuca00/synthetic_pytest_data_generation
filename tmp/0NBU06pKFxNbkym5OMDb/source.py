def ensure_ref_position_is_valid(ref_position, num_alts, param_title):
    
    assert param_title in ['intercept_names', 'shape_names']

    try:
        assert ref_position is None or isinstance(ref_position, int)
    except AssertionError:
        msg = "ref_position for {} must be an int or None."
        raise TypeError(msg.format(param_title))

    if param_title == "intercept_names":
        try:
            assert ref_position is not None
        except AssertionError:
            raise ValueError("At least one intercept should be constrained.")

    try:
        if ref_position is not None:
            assert ref_position >= 0 and ref_position <= num_alts - 1
    except AssertionError:
        msg = "ref_position must be between 0 and num_alts - 1."
        raise ValueError(msg)

    return None
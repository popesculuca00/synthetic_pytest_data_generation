def _check_components(obj, components=None, check_size=True, valid_sizes=[2, 3]):
    
    try:
        if check_size and (obj.columns.size not in valid_sizes):
            assert len(components) in valid_sizes

        if components is None:
            components = obj.columns.values
    except:
        msg = "Suggest components or provide a slice of the dataframe."
        raise AssertionError(msg)
    return components
def bbknn(adata, batch_key='batch', copy=False, **kwargs):
    
    params = locals()  # Has to be first
    kwargs = params.pop('kwargs')
    try:
        from bbknn import bbknn
    except ImportError:
        raise ImportError('Please install bbknn: `pip install bbknn`.')
    return bbknn(**params, **kwargs)
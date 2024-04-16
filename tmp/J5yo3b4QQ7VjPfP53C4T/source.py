def _tracer_bias_beta(params, name):
    
    growth_rate = params.get("growth_rate", 1.)

    bias = params.get('bias_' + name, None)
    bias_eta = params.get('bias_eta_' + name, None)
    beta = params.get('beta_' + name, None)

    err_msg = ("For each tracer, you need to specify two of these three:"
               " (bias, bias_eta, beta)."
               " If all three are given, we use bias and beta.")

    if bias is None:
        assert bias_eta is not None and beta is not None, err_msg
        bias = bias_eta * growth_rate / beta

    if bias_eta is None:
        assert bias is not None and beta is not None, err_msg

    if beta is None:
        assert bias is not None and bias_eta is not None, err_msg
        beta = bias_eta * growth_rate / bias

    return bias, beta
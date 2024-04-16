
def lightness_correlate(Y_b, Y_w, Q, Q_w):
    Z = 1 + (Y_b / Y_w) ** 0.5
    J = 100 * (Q / Q_w) ** Z
    return J

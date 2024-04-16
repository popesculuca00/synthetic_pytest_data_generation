def mmd2(PPk, QQk, PQk):
    
    assert(PQk is not None)
    # Allow `PPk` to be None, if we want to compute mmd2 for the generator
    if PPk is None:
        PPk_ = 0
    elif len(PPk.shape) == 2:
        m = PPk.size(0)
        PPk_ = (PPk.sum() - PPk.trace()) / (m**2 - m) if m != 1 else 0
    elif len(PPk.shape) == 1:
        PPk_ = PPk.mean()
    elif len(PPk.shape) == 0:
        PPk_ = PPk
    else:
        raise ValueError("Not supported `PPk`.")

    if QQk is None:
        QQk_ = 0
    elif len(QQk.shape) == 2:
        n = QQk.size(0)
        QQk_ = (QQk.sum() - QQk.trace()) / (n**2 - n) if n != 1 else 0
    elif len(QQk.shape) == 1:
        QQk_ = QQk.mean()
    elif len(QQk.shape) == 0:
        QQk_ = QQk
    else:
        raise ValueError("Not supported `QQk`.")

    if PQk.size():
        PQk_ = PQk.mean()
    else:
        PQk_ = PQk

    return PPk_ + QQk_ - 2 * PQk_
def tvd(x0, rho, gamma):
    
    try:
        from skimage.restoration import denoise_tv_bregman
    except ImportError:
        print('Error: scikit-image not found. TVD will not work.')
        return x0

    return denoise_tv_bregman(x0, rho / gamma)
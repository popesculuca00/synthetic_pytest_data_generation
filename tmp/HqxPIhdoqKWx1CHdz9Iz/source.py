def augmented_image_transforms(d=0, t=0, s=0, sh=0, ph=0, pv=0, resample=2):
    r
    from torchvision import transforms

    degrees = d
    translate = None if t == 0 else (t, t)
    scale = None if s == 0 else (1 - s, 1 + s)
    shear = None if sh == 0 else sh
    return transforms.Compose([transforms.RandomAffine(degrees, translate, scale, shear, resample),
                             transforms.RandomHorizontalFlip(ph),
                             transforms.RandomVerticalFlip(pv),
                             transforms.ToTensor(),
                             transforms.Normalize(*[[0.5] * 3] * 2)])
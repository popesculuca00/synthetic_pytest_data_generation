def geometrical_spreading(freq, dist, model="REA99"):
    

    if model == "REA99":
        dist_cross = 40.0
        if dist <= dist_cross:
            geom = dist ** (-1.0)
        else:
            geom = (dist * dist_cross) ** (-0.5)
    else:
        raise ValueError("Unsupported anelastic attenuation model.")
    return geom
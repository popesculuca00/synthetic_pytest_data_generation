def serialize_skycoord(o):
    
    representation = o.representation.get_name()
    frame = o.frame.name

    r = o.represent_as('spherical')

    d = dict(
        _type='astropy.coordinates.SkyCoord',
        frame=frame,
        representation=representation,
        lon=r.lon,
        lat=r.lat)

    if len(o.distance.unit.to_string()):
        d['distance'] = r.distance

    return d
def bounding_box_from(points, i, i1, thr, debug = False):
    
    pi = points[i]
    pi1 = points[i1]

    min_lat = min(pi.lat, pi1.lat)
    min_lon = min(pi.lon, pi1.lon)
    max_lat = max(pi.lat, pi1.lat)
    max_lon = max(pi.lon, pi1.lon)

    return min_lat-thr, min_lon-thr, max_lat+thr, max_lon+thr
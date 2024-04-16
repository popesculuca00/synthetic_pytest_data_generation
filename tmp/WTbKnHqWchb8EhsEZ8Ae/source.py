def in_hull(points, hull):
    
    from scipy.spatial import Delaunay, ConvexHull
    if isinstance(hull, ConvexHull):
        hull = hull.points
    hull = Delaunay(hull)
    return hull.find_simplex(points) >= 0
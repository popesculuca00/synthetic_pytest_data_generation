import numpy

def qcriterion(velocity, grid):
    
    u, v, w = velocity
    x, y, z = grid

    dudx = numpy.gradient(u, x, axis=2)
    dudy = numpy.gradient(u, y, axis=1)
    dudz = numpy.gradient(u, z, axis=0)

    dvdx = numpy.gradient(v, x, axis=2)
    dvdy = numpy.gradient(v, y, axis=1)
    dvdz = numpy.gradient(v, z, axis=0)

    dwdx = numpy.gradient(w, x, axis=2)
    dwdy = numpy.gradient(w, y, axis=1)
    dwdz = numpy.gradient(w, z, axis=0)

    qcrit = (-0.5 * (dudx**2 + dvdy**2 + dwdz**2) -
             dudy * dvdx - dudz * dwdx - dvdz * dwdy)

    return qcrit
import numpy

def dc(input1, input2):
    
    input1 = numpy.atleast_1d(input1.astype(numpy.bool))
    input2 = numpy.atleast_1d(input2.astype(numpy.bool))

    intersection = numpy.count_nonzero(input1 & input2)

    size_i1 = numpy.count_nonzero(input1)
    size_i2 = numpy.count_nonzero(input2)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc
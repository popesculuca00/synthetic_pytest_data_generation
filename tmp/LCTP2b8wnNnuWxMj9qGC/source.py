import numpy

def jc(input1, input2):
    
    input1 = numpy.atleast_1d(input1.astype(numpy.bool))
    input2 = numpy.atleast_1d(input2.astype(numpy.bool))

    intersection = numpy.count_nonzero(input1 & input2)
    union = numpy.count_nonzero(input1 | input2)

    jc = float(intersection) / float(union)

    return jc
def matrix_indices(vector_idx, matrix_size):
    

    assert vector_idx < matrix_size * (matrix_size + 1) / 2, 'Invalid vector_idx for this matrix_size'

    # Work out which diagonal the element is on and its index on that diagonal, by iterating over the diagonals
    diag_length = matrix_size
    while vector_idx - diag_length >= 0:
        vector_idx -= diag_length
        diag_length -= 1
    diag = matrix_size - diag_length

    # Index at the top of the diagonal is (row = 0, col = diag),
    # so index of element is (row = vector_idx, col = diag + vector_idx)
    row = vector_idx
    col = diag + vector_idx
    return row, col
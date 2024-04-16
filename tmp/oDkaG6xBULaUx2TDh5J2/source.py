def batch_mat_vec(sparse_matrix, vector_batch):
    

    # (b, n) -> (n, b)
    matrices = vector_batch.transpose(0, 1)

    # (k, b) -> (b, k)
    return sparse_matrix.mm(matrices).transpose(1, 0)
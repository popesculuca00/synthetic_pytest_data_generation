import torch

def _pairwise_distances(embeddings, squared=False):
    
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1)) # embeddings.T
 
    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = dot_product.diag()

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)


    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances.relu_() # distances = distances.relu()

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances.sqrt_()

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances
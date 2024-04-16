import torch

def hard_examples_mining(dist_mat, identity_mat, return_idxes=False):
    r
    # the implementation here is a little tricky, dist_mat contains pairwise distance between probe image and other
    # images in current mini-batch. As we want to select positive examples of the same person, we add a constant
    # negative offset on other images before sorting. As a result, images of the **same** person will rank first.
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + (-1e7) * (1 - identity_mat), dim=1,
                                               descending=True)
    dist_ap = sorted_dist_mat[:, 0]
    hard_positive_idxes = sorted_idxes[:, 0]

    # the implementation here is similar to above code, we add a constant positive offset on images of same person
    # before sorting. Besides, we sort in ascending order. As a result, images of **different** persons will rank first.
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + 1e7 * identity_mat, dim=1,
                                               descending=False)
    dist_an = sorted_dist_mat[:, 0]
    hard_negative_idxes = sorted_idxes[:, 0]
    if return_idxes:
        return dist_ap, dist_an, hard_positive_idxes, hard_negative_idxes
    return dist_ap, dist_an
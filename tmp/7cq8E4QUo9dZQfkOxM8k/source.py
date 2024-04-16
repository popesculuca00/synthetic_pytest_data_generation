def compute_prior_probability(alpha):
    
    prior_leaf_prob = [0]
    depth = 1
    while prior_leaf_prob[-1] < 1:
        prior_leaf_prob.append(1 - alpha ** depth)
        depth += 1
    return prior_leaf_prob
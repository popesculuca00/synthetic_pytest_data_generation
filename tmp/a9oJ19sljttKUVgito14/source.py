def transformed_potential_energy(potential_energy, inv_transform, z):
    
    x, intermediates = inv_transform.call_with_intermediates(z)
    logdet = inv_transform.log_abs_det_jacobian(z, x, intermediates=intermediates)
    return potential_energy(x) - logdet
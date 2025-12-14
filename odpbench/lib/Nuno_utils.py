import numpy as np

def get_nuno(test_probs):
    singular_values = np.linalg.svd(test_probs, compute_uv=False)
    nuclear_norm = np.sum(singular_values)
    min_dim = min(test_probs.shape[0], test_probs.shape[1])

    return nuclear_norm / np.sqrt(min_dim * test_probs.shape[0])
import numpy as np

def get_DoC(train_probs, test_probs): 
    # Difference Between AC (Average Confidence) of Train and Test
    train_max_probs = np.max(train_probs, axis=-1)
    test_max_probs = np.max(test_probs, axis=-1)

    return np.mean(train_max_probs) - np.mean(test_max_probs)

def get_DoE(train_probs, test_probs):
    train_entropy = - train_probs * np.log(train_probs)
    test_entropy = - test_probs * np.log(test_probs)

    train_entropy = np.mean(np.sum(train_entropy, axis=1))
    test_entropy = np.mean(np.sum(test_entropy, axis=1))
    
    return np.mean(train_entropy) - np.mean(test_entropy)
"""Create last figure of the paper."""
import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import norm

################
# Gathering data
################
def is_pretrained(rule):
    """TODO: Update this definition as other model families, e.g. ODST get integrated"""
    if hasattr(rule.params, "pretrained") and rule.params.pretrained:
        return True
    if "clip" in rule.db_obj.subtype:
        return True
    return False

def fetch_trajectory():
    from scipy.special import softmax
    N = []
    NV2 = []
    L = []
    LV2 = []
    dir = 'resnet18_seed0_pretrained_False'
    for file in tqdm(os.listdir(dir)):
        
        if 'L_test' in file:
            L.append(np.load(f'{dir}/' + file))
            N.append(int(file.split('_')[2].split('.')[0]))
            
        if 'L_target' in file:
            LV2.append(np.load(f'{dir}/' + file))
            NV2.append(int(file.split('_')[2].split('.')[0]))

    T = np.load('Y_cifar10.npy')
    Tv2 = np.load('Y_cifar10.1.npy')
    
    idx = np.argsort(N)
    L = [L[i] for i in idx]
    N = [N[i] for i in idx]
    print(N)

    idx = np.argsort(NV2)
    LV2 = [LV2[i] for i in idx]
    NV2 = [NV2[i] for i in idx]

    evaluations=[]
    for i in tqdm(range(len(L))):
        L1 = L[i]
        L2 = LV2[i]
        evaluations.append(dict(
            test_accuracy=np.mean(np.argmax(L1, axis=-1) == T),
            shift_accuracy=np.mean(np.argmax(L2, axis=-1) == Tv2),
            test_pred=np.argmax(L1, axis=-1),
            shift_pred=np.argmax(L2, axis=-1)))

    return pd.DataFrame(evaluations)

def rescale(data, scaling=None):
    """Rescale the data."""
    data = np.asarray(data)
    if scaling == "probit":
        return norm.ppf(data)
    elif scaling == "logit":
        return np.log(data / (1 - data))
    elif scaling == "linear":
        return data
    raise NotImplementedError

def compute_linear_fit(x, y):
    """Returns bias and slope from regression y on x."""
    x = np.array(x)
    y = np.array(y)

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared
    
def aline(test_preds, test_accuracy, shift_preds):
    test_accuracy = rescale(test_accuracy, 'probit')
    A = []
    test_agrs = []
    shift_agrs = []
    test_accs = []
    n = len(test_preds)
    for i in range(n):
        for j in range(i, n):
            a = np.zeros(n)
            a[i] = 0.5
            a[j] = 0.5
            A.append(a)
            test_agrs.append(np.mean(test_preds[i] == test_preds[j]))
            shift_agrs.append(np.mean(shift_preds[i] == shift_preds[j]))
            test_accs.append(0.5*test_accuracy[i] + 0.5*test_accuracy[j])
    A, test_agrs, shift_agrs, test_accs = np.array(A), np.array(test_agrs), np.array(shift_agrs), np.array(test_accs)
    A, test_agrs, shift_agrs, test_accs= A[test_agrs <= 0.98], test_agrs[test_agrs <= 0.98], shift_agrs[test_agrs <= 0.98], test_accs[test_agrs <= 0.98]
    A, test_agrs, shift_agrs, test_accs = A[test_agrs >= 0.05], test_agrs[test_agrs >= 0.05], shift_agrs[test_agrs >= 0.05], test_accs[test_agrs >= 0.05]
    A, test_agrs, shift_agrs, test_accs = A[shift_agrs <= 0.98], test_agrs[shift_agrs <= 0.98], shift_agrs[shift_agrs <= 0.98], test_accs[shift_agrs <= 0.98]
    A, test_agrs, shift_agrs, test_accs = A[shift_agrs >= 0.05], test_agrs[shift_agrs >= 0.05], shift_agrs[shift_agrs >= 0.05], test_accs[shift_agrs >= 0.05]
    test_agrs, shift_agrs = rescale(test_agrs, 'probit'), rescale(shift_agrs, 'probit')
    (bias, slope), r2 = compute_linear_fit(test_agrs, shift_agrs)
    b = shift_agrs + slope * (test_accs - test_agrs)
    print('linear fit', bias, slope)
    w, _, _, _ = np.linalg.lstsq(A, b)
    pred_s = slope * test_accuracy + bias
    pred_d = norm.cdf(w)
    return (pred_s, pred_d), bias, slope

def ac(shift_preds):
    return np.mean(max_confidence(shift_preds))

def doc(test_preds, test_accuracy, shift_preds):
    return test_accuracy + (ac(shift_preds) - ac(test_preds))

def negative_entropy(test_preds):
    return np.sum(test_preds*np.log(test_preds), axis=-1)
    
def max_confidence(test_preds):
    return  np.max(test_preds, axis=1)

def mae(preds, acc):
    return np.mean(np.abs(preds - acc))

def get_predictions(evals):
    (aline_s, aline_d), _, _ = aline(evals.test_pred, evals['test_accuracy'], evals.shift_pred)
    return aline_s, aline_d


from scipy.stats import norm

def Agreement_On_line(evals):
    aline_s, aline_d = get_predictions(evals)
    return aline_d
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=1)
    ax.plot(np.arange(0,150,5), evals['shift_accuracy'].values, label='OOD Accuracy', color='tab:blue', linewidth=2)
    ax.plot(np.arange(0,150,5), aline_d, label='ALine-D', color='tab:pink', linewidth=2)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.legend(loc='lower right', prop={'size': 12})
    ax.set_yticklabels([f"{int(100. * loc)}" for loc in [0.1, 0.3, 0.5, 0.7, 0.9]])
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.suptitle('Along One Trajectory', fontsize=17)
    print(mae(aline_d, evals['shift_accuracy'].values))

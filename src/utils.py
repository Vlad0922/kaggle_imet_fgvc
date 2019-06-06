import random
import os
import numpy as np
import torch
from collections import Counter, defaultdict

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def sigmoid(x):
    return 1/(1+np.exp(-x))


def create_oh_labels(df, nclasses):
    res = np.zeros((df.shape[0], nclasses))
    
    for idx, lbls in enumerate(df.attribute_ids.values):
        res[idx, lbls] = 1
    
    return res


# copied from Konstatin Lopuhin github
# https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/make_folds.py
def make_folds(df, n_folds: int) -> pd.DataFrame:
    cls_counts = Counter(cls for classes in df['attribute_ids']
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in df.sample(frac=1, random_state=42).itertuples():
        cls = min(item.attribute_ids, key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids:
            fold_cls_counts[fold, cls] += 1
            
    return np.array(folds)

def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask
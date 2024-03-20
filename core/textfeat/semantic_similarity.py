import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.load import load_data 
import torch
from scipy.spatial import distance
import numpy as np 

distance_metric = {
    # 'cos': distance.cosine,
    # 'l2': distance.euclidean,
    # 'hamming': distance.hamming,
    # 'jaccard': distance.jaccard,
    # 'dice': distance.dice,
    'dot': lambda x, y: np.dot(x, y)
}

def pairwise_prediction(data, test_index, distance):
    """predict the link existence using the pairwise textual similarity"""
    test_index = test_index.numpy().transpose()
    test_pred = []

    if distance not in distance_metric:
        raise ValueError("Invalid distance metric specified.")

    metric_function = distance_metric[distance]

    for ind in test_index:
        metric_value = metric_function(data[ind[0]], data[ind[1]])
        test_pred.append(metric_value)

    test_pred = torch.tensor(np.asarray(test_pred))
    return test_pred

def compat_matrix(data, test_index, ):
    return 

# test_pos_pred, test_neg_pred = get_fp_prediction(data_embeddings, test_pos, test_neg, args)
# valid_pos_pred, valid_neg_pred = get_fp_prediction(data_embeddings, valid_pos, valid_neg, args)

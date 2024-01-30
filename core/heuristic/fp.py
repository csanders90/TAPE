import torch
from scipy.spatial import distance
import numpy as np 


def get_fp_prediction(data, test_pos, test_neg, distance):
    test_pos, test_neg = test_pos.numpy().transpose(), test_neg.numpy().transpose()
    test_pos_pred, test_neg_pred = [], []

    distance_metric = {
        'cos': distance.cosine,
        'l2': distance.euclidean,
        'hamming': distance.hamming,
        'jaccard': distance.jaccard,
        'dice': distance.dice,
        'dot': lambda x, y: np.dot(x, y)
    }

    if distance not in distance_metric:
        raise ValueError("Invalid distance metric specified.")

    metric_function = distance_metric[distance]

    for ind in test_pos:
        metric_value = metric_function(data[ind[0]], data[ind[1]])
        test_pos_pred.append(metric_value)

    for ind_n in test_neg:
        metric_value = metric_function(data[ind_n[0]], data[ind_n[1]])
        test_neg_pred.append(metric_value)

    test_pos_pred, test_neg_pred = torch.tensor(np.asarray(test_pos_pred)), torch.tensor(np.asarray(test_neg_pred))
    return test_pos_pred, test_neg_pred


# test_pos_pred, test_neg_pred = get_fp_prediction(data_embeddings, test_pos, test_neg, args)
# valid_pos_pred, valid_neg_pred = get_fp_prediction(data_embeddings, valid_pos, valid_neg, args)
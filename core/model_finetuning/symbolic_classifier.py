import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import numpy as np
import torch
# from sklearn.ensemble import RandomForestClassifier
import cuml

from cupy import asnumpy
from joblib import dump, load
from cuml.datasets.classification import make_classification
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.model_selection import train_test_split, GridSearchCV

# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp
import optuna
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import Evaluator
from sklearn.metrics import accuracy_score

# embedding_model_name = "tfidf"

# # Load datasets
# train_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_train_dataset.pt').numpy()
# train_labels = torch.load(f'./generated_dataset/{embedding_model_name}_train_labels.pt').numpy()
# val_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_val_dataset.pt').numpy()
# val_labels = torch.load(f'./generated_dataset/{embedding_model_name}_val_labels.pt').numpy()
# test_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_test_dataset.pt').numpy()
# test_labels = torch.load(f'./generated_dataset/{embedding_model_name}_test_labels.pt').numpy()

# # Combine train and validation datasets for cross-validation
# X_train = np.concatenate((train_dataset, val_dataset), axis=0)
# y_train = np.concatenate((train_labels, val_labels), axis=0)

# """# Define the parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize the RandomForestClassifier
# rf = RandomForestClassifier()

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print(f'Best hyperparameters: {best_params}')

# # Train the final model with the best hyperparameters
# best_rf = RandomForestClassifier(**best_params)"""
# clf = cuRF(n_estimators=100, max_depth=10, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate on the test set
# y_pred = clf.predict(test_dataset)
# y_pred_proba = clf.predict_proba(test_dataset)[:, 1]

# # Print evaluation metrics
# print(f'Accuracy: {accuracy_score(test_labels, y_pred):.4f}')
# print(f'ROC AUC: {roc_auc_score(test_labels, y_pred_proba):.4f}')
# print('Confusion Matrix:')
# print(confusion_matrix(test_labels, y_pred))

# synthetic dataset dimensions
n_samples = 1000
n_features = 10
n_classes = 2

# random forest depth and size
n_estimators = 25
max_depth = 10

# generate synthetic data [ binary classification task ]
X, y = make_classification ( n_classes = n_classes,
                             n_features = n_features,
                             n_samples = n_samples,
                             random_state = 0 )

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 0 )

model = cuRF( max_depth = max_depth,
              n_estimators = n_estimators,
              random_state  = 0 )

trained_RF = model.fit ( X_train, y_train )

predictions = model.predict ( X_test )

cu_score = cuml.metrics.accuracy_score( y_test, predictions )
sk_score = accuracy_score( asnumpy( y_test ), asnumpy( predictions ) )

print( " cuml accuracy: ", cu_score )
print( " sklearn accuracy : ", sk_score )
import pickle

single_gpu_model = trained_RF.get_combined_model()
pickle.dump(single_gpu_model, open("kmeans_model.pkl", "wb"))
single_gpu_model = pickle.load(open("kmeans_model.pkl", "rb"))
single_gpu_model.cluster_centers_
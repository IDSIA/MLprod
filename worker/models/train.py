from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

from worker.models.model import Model

import pandas as pd
import joblib
import json
import numpy as np
import os


def train_model(
        dataset: pd.DataFrame,
        path: str='./models',
        k_best: int=20,
        epochs: int=100,
        batch_size: int=8,
        frac1: float=0.5,
        random_state: int=42,
        metrics_list: list[str]|None=None
    ) -> dict[str, dict[str, list[float]]]:
    """Train the model. If required it can also evaluate the model against a test set.

    :param df_tr:
        Pandas' DataFrame for training.
    :param mms_path:
        Path to store the MinMaxScaler model (default: ./models/mms.model). 
    :param skb_path:
        Path to store the SelectKBest model (default ./models/skb.model).
    :param model_path:
        Path to store the trained model (default: ./models/neuralnet.model).
    :param best_model_path:
        Path to store the model with the best loss (default: ./models/best_nn.torch.model).
    :param k_best:
        Number of features to extract with SelectKBest algorithm (default: 20).
    :param epochs:
        Number of epochs to run during training (default: 100).
    :param batch_size:
        Size of the mini-batches (default: 8).
    :param frac1: 
        Proportion of the labels equal to 1 (default: 0.5 which mean same quantity as 0).
    :param random_state:
        Seed for random generation (default: 42).
    :param save_best:
        If this flag is true, the best model will be saved to the best_model_path location (default: False).
    :param metrics_list: 
        List of metrics to check for evaluation, also with the test set if availble. (Default: None, which means no metrics except Loss will be tracked).
    
    :return:
        A dictionary with a list of results for each tracked metric.
    """
    path_mms: str = str(os.path.join(path, 'mms.model'))
    path_skb: str = str(os.path.join(path, 'skb.model'))
    path_model: str = str(os.path.join(path, 'neuralnet.model'))
    path_metadata: str = str(os.path.join(path, 'metadata.json'))

    X = dataset.drop('label', axis=1).values
    Y = dataset['label'].values

    n_records, x_input = X.shape

    # Preprocessing: MinMaxScaler ---------------------------------------------
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    joblib.dump(mms, path_mms)

    # Preprocessing: FeatureSelection -----------------------------------------
    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X, Y)
    X = skb.transform(X)

    joblib.dump(skb, path_skb)

    x_output = X.shape[1]

    # Training: setup ---------------------------------------------------------
    n, features = X.shape
    batch_count = int(n / batch_size)
    r = np.random.default_rng(random_state)

    model = Model(features).to('cpu')

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    metrics = {
        'train': {
            'loss': []
        },
        'test': {
            'loss': []
        }
    }

    if metrics_list is not None:
        for metric in metrics_list:
            if metric == 'accuracy':
                metrics['train']['acc'] = []
                metrics['test']['acc'] = []
            if metric == 'precision':
                metrics['train']['pre'] = []
                metrics['test']['pre'] = []
            if metric == 'recall':
                metrics['train']['rec'] = []
                metrics['test']['rec'] = []
            if metric == 'f1':
                metrics['train']['f1'] = []
                metrics['test']['f1'] = []
            if metric == 'auc':
                metrics['train']['auc'] = []
                metrics['test']['auc'] = []

    # Training: run -----------------------------------------------------------
    for _ in range(epochs):
        # train
        model.train()

        loss_btc, y_preds, y_trues = [], [], []
        for _ in range(batch_count):
            mask = (Y == 0).reshape(-1)

            X_tr_0 = X[mask]
            X_tr_1 = X[~mask]

            Y_tr_0 = Y[mask]
            Y_tr_1 = Y[~mask]

            n0, _ = X_tr_0.shape
            n1, _ = X_tr_1.shape

            sample_ids_0 = r.choice(n0, size=min(n, int(batch_size * (1 - frac1))))
            sample_ids_1 = r.choice(n1, size=min(n, int(batch_size * frac1)))

            x_tr = np.vstack((X_tr_0[sample_ids_0], X_tr_1[sample_ids_1]))
            y_tr = np.vstack((Y_tr_0[sample_ids_0], Y_tr_1[sample_ids_1]))

            x = torch.FloatTensor(x_tr).to('cpu')
            y = torch.FloatTensor(y_tr).to('cpu')

            out = model(x)
            loss = criterion(out, y)
            l_val = loss.item()
            
            loss_btc.append(l_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_preds.append(out.detach().numpy())
            y_trues.append(y_tr)

    y_preds = np.array(y_preds).reshape(-1)
    y_trues = np.array(y_trues).reshape(-1)

    # Training: record metrics --------------------------------------------
    loss_btc_mean = np.array(loss_btc).mean()
    metrics['train']['loss'].append(loss_btc_mean)
        
    evaluate(y_trues, y_preds, metrics['train'])

    torch.save(model.state_dict(), path_model)

    with open(path_metadata, 'w+') as f:
        json.dump({
            'features': dataset.drop('label', axis=1).columns.to_list(),
            'x_input':  x_input,
            'x_output': x_output,
            'n_records': n_records,
            'seed': random_state,
        }, f, indent=4)

    return metrics


def evaluate(y_trues, y_preds, metrics: dict[str, list[float]], pred_threshold: float=0.5):
    if 'auc' in metrics:
        metrics['auc'].append(roc_auc_score(y_trues, y_preds))

    y_preds = (y_preds > pred_threshold).astype('int')

    if 'acc' in metrics:
        metrics['acc'].append(accuracy_score(y_trues, y_preds))
    if 'pre' in metrics:
        metrics['pre'].append(recall_score(y_trues, y_preds, zero_division=0))
    if 'rec' in metrics:
        metrics['rec'].append(precision_score(y_trues, y_preds, zero_division=0))
    if 'f1' in metrics:
        metrics['f1'].append(f1_score(y_trues, y_preds, zero_division=0))

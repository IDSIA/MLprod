from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

from worker.models.model import Model

import pandas as pd
import joblib
import json
import logging
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
    Y = dataset['label'].values.reshape(-1, 1)

    n_records, x_input = X.shape

    # Preprocessing: MinMaxScaler ---------------------------------------------
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    joblib.dump(mms, path_mms)

    logging.info(f'training: MinMaxScaler saved to {path_mms}')

    # Preprocessing: FeatureSelection -----------------------------------------
    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X, Y)
    X = skb.transform(X)

    joblib.dump(skb, path_skb)

    logging.info(f'training: SelectKBest saved to {path_skb}')

    # Training: setup ---------------------------------------------------------
    n, x_output = X.shape

    batch_count = int(n / batch_size)
    r = np.random.default_rng(random_state)

    logging.info(f'training: creating model with input {x_output}')

    model = Model(x_output).to('cpu')

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    metrics = {
        'loss': []
    }

    # Training: run -----------------------------------------------------------
    mask = (Y == 0).reshape(-1)

    X_tr_0 = X[mask]
    X_tr_1 = X[~mask]

    Y_tr_0 = Y[mask]
    Y_tr_1 = Y[~mask]

    n0, _ = X_tr_0.shape
    n1, _ = X_tr_1.shape

    batch0_size = min(n, int(batch_size * (1 - frac1)))
    batch1_size = min(n, int(batch_size * frac1))

    for epoch in range(epochs):
        logging.info(f'training: epoch {epoch}/{epochs}')

        # train
        model.train()

        loss_btc, y_preds, y_trues = [], [], []
        for _ in range(batch_count):
            sample_ids_0 = r.choice(n0, size=batch0_size)
            sample_ids_1 = r.choice(n1, size=batch1_size)

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

    logging.info(f'training: completed')

    y_preds = np.array(y_preds).reshape(-1)
    y_trues = np.array(y_trues).reshape(-1)

    # Training: record metrics --------------------------------------------
    metrics = {}

    loss_btc_mean = np.array(loss_btc).mean()
    metrics['loss'] = loss_btc_mean
        
    metrics = metrics | evaluate(y_trues, y_preds, metrics_list)

    for k,v in metrics.items():
        logging.info(f'train metric {k}: {v:.4}')

    torch.save(model.state_dict(), path_model)

    logging.info(f'training: model saved to {path_model}')

    with open(path_metadata, 'w+') as f:
        json.dump({
            'features': dataset.drop('label', axis=1).columns.to_list(),
            'x_input':  x_input,
            'x_output': x_output,
            'n_records': n_records,
            'seed': random_state,
        }, f, indent=4)

    logging.info(f'training: metadata saved to {path_metadata}')

    return metrics


def evaluate(y_trues, y_preds, metrics_list: list[str], pred_threshold: float=0.5) -> dict[str, list[float]]:
    metrics = dict()

    if 'auc' in metrics_list:
        metrics['auc'] = roc_auc_score(y_trues, y_preds)

    y_preds = (y_preds > pred_threshold).astype('int')

    if 'acc' in metrics_list:
        metrics['acc'] = accuracy_score(y_trues, y_preds)
    if 'pre' in metrics_list:
        metrics['pre'] = recall_score(y_trues, y_preds, zero_division=0)
    if 'rec' in metrics_list:
        metrics['rec'] = precision_score(y_trues, y_preds, zero_division=0)
    if 'f1' in metrics_list:
        metrics['f1'] = f1_score(y_trues, y_preds, zero_division=0)
    
    return metrics
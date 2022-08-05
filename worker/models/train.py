import joblib
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

from worker.models.model import Model

import os


def train_model(
        X_tr, 
        Y_tr, 
        X_ts=None,
        Y_ts=None, 
        path: str='./models',
        k_best: int=20,
        epochs: int=100,
        batch_size: int=8,
        frac1: float=0.5,
        random_state: int=42,
        metrics_list: list[str]|None=None
    ) -> dict[str, dict[str, list[float]]]:
    """Train the model. If required it can also evaluate the model against a test set.

    :param X_tr:
        Train data.
    :param Y_tr:
        Label for train data.
    :param X_ts:
        Test data (Optional, if missing no evaluation will be done).
    :param Y_ts:
        Label for the test data (Optional, if missing no evaluation will be done).
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
    mms_path: str = str(os.path.join(path, 'mms.model'))
    skb_path: str = str(os.path.join(path, 'skb.model'))
    model_path: str = str(os.path.join(path, 'neuralnet.model'))

    # Preprocessing: MinMaxScaler ---------------------------------------------
    mms = MinMaxScaler()
    X_tr = mms.fit_transform(X_tr)
    if X_ts is not None:
        X_ts = mms.transform(X_ts)

    joblib.dump(mms, mms_path)

    # Preprocessing: FeatureSelection -----------------------------------------
    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X_tr, Y_tr)
    X_tr = skb.transform(X_tr)
    if X_ts is not None:
        X_ts = skb.transform(X_ts)

    joblib.dump(skb, skb_path)

    # Training: setup ---------------------------------------------------------
    n, features = X_tr.shape
    batch_count = int(n / batch_size)
    r = np.random.default_rng(random_state)

    model = Model(features).to('cpu')

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    if X_ts is not None and Y_ts is not None:
        x_ts = torch.FloatTensor(X_ts).to('cpu')
        y_ts = torch.FloatTensor(Y_ts).to('cpu')

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
            mask = (Y_tr == 0).reshape(-1)

            X_tr_0 = X_tr[mask]
            X_tr_1 = X_tr[~mask]

            Y_tr_0 = Y_tr[mask]
            Y_tr_1 = Y_tr[~mask]

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
        
    evaluate(y_trues, y_preds, metrics['test'])

    if X_ts is None and Y_ts is None:
        model.eval()

        # Training: evaluation ------------------------------------------------
        out = model(x_ts)

        loss = criterion(out, y_ts)
        l_val = loss.item()

        metrics['test']['loss'].append(l_val)

        pred_ts = out.detach().numpy()

        evaluate(Y_ts, pred_ts, metrics['test'])

    torch.save(model.state_dict(), model_path)

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

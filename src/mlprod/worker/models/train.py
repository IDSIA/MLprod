from pathlib import Path
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from mlprod.worker.models.model import Model

import torch
import torch.nn as nn

import pandas as pd
import joblib
import json
import logging
import numpy as np

LOGGER = logging.getLogger("mlprod.worker.models.train")

DEFAULT_MODELS_DIR = Path("./models")


def train_model(
    dataset: pd.DataFrame,
    path: Path = DEFAULT_MODELS_DIR,
    k_best: int = 20,
    epochs: int = 100,
    batch_size: int = 8,
    frac1: float = 0.5,
    random_state: int = 42,
    metrics_list: list[str] = list(),
) -> dict[str, dict[str, list[float]]]:
    """Train the model. If required it can also evaluate the model against a test set.

    :param dataset:
        Pandas' DataFrame for training.
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
    :param metrics_list:
        List of metrics to check for evaluation, also with the test set if availble. (Default: None, which means no metrics except Loss will be tracked).
        Possible values are `auc` (ROC AUC curve), `acc` (accuracy), `pre` (Precision), `rec` (Recall), `f1` (f1 score).

    :return:
        A dictionary with a list of results for each tracked metric.
    """
    path_mms: Path = path / "mms.model"
    path_skb: Path = path / "skb.model"
    path_model: Path = path / "neuralnet.model"
    path_metadata: Path = path / "metadata.json"

    X = dataset.drop("label", axis=1).values
    Y = dataset["label"].values.reshape(-1, 1)  # type: ignore

    n_records, x_input = X.shape

    # Preprocessing: MinMaxScaler ---------------------------------------------
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    joblib.dump(mms, path_mms)

    LOGGER.info(f"training: MinMaxScaler saved to {path_mms}")

    # Preprocessing: FeatureSelection -----------------------------------------
    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X, Y)
    X = skb.transform(X)

    joblib.dump(skb, path_skb)

    LOGGER.info(f"training: SelectKBest saved to {path_skb}")

    # Training: setup ---------------------------------------------------------
    if X.shape is None:
        raise ValueError("Shape of X input value is none!")

    n, x_output = X.shape

    batch_count = int(n / batch_size)
    r = np.random.default_rng(random_state)

    LOGGER.info(f"training: creating model with input {x_output}")

    model = Model(x_output).to("cpu")

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    metrics = {"loss": []}

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
        LOGGER.info(f"training: epoch {epoch}/{epochs}")

        # train
        model.train()

        loss_btc, y_preds, y_trues = [], [], []
        for _ in range(batch_count):
            sample_ids_0 = r.choice(n0, size=batch0_size)
            sample_ids_1 = r.choice(n1, size=batch1_size)

            x_tr = np.vstack((X_tr_0[sample_ids_0], X_tr_1[sample_ids_1]))
            y_tr = np.vstack((Y_tr_0[sample_ids_0], Y_tr_1[sample_ids_1]))

            x = torch.FloatTensor(x_tr).to("cpu")
            y = torch.FloatTensor(y_tr).to("cpu")

            out = model(x)

            loss = criterion(out, y)
            l_val = loss.item()

            loss_btc.append(l_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_preds.append(out.detach().numpy())
            y_trues.append(y_tr)

    LOGGER.info("training: completed")

    y_preds = np.array(y_preds).reshape(-1)
    y_trues = np.array(y_trues).reshape(-1)

    # Training: record metrics --------------------------------------------
    metrics = {}

    loss_btc_mean = np.array(loss_btc).mean()
    metrics["loss"] = loss_btc_mean

    metrics = metrics | evaluate(y_trues, y_preds, metrics_list)

    for k, v in metrics.items():
        LOGGER.info(f"train metric {k}: {v:.4}")

    torch.save(model.state_dict(), path_model)

    LOGGER.info(f"training: model saved to {path_model}")

    with open(path_metadata, "w+") as f:
        json.dump(
            {
                "features": dataset.drop("label", axis=1).columns.to_list(),
                "x_input": x_input,
                "x_output": x_output,
                "n_records": n_records,
                "seed": random_state,
            },
            f,
            indent=4,
        )

    LOGGER.info(f"training: metadata saved to {path_metadata}")

    return metrics


def evaluate(
    y_trues, y_preds, metrics_list: list[str], pred_threshold: float = 0.5
) -> dict[str, float]:
    """Evaluate the performance over a list of given metrics.

    :param y_trues:
        True values to test against.
    :param y_preds:
        Direct output values of the model (discretization is done internally).
    :param metrics_list:
        List of metrics to track. Possible values are `auc` (ROC AUC curve), `acc` (accuracy),
        `pre` (Precision), `rec` (Recall), `f1` (f1 score).
    :param pred_threshold:
        Threshold for class definition: 1 above this threshold, otherwise class 0.

    :return:
        A dictionary with the metric value for each metric entry in the `metrics_list` argument.
    """
    metrics = dict()

    if "auc" in metrics_list:
        metrics["auc"] = roc_auc_score(y_trues, y_preds)

    # discretize
    y_preds = (y_preds > pred_threshold).astype("int")

    if "acc" in metrics_list:
        metrics["acc"] = accuracy_score(y_trues, y_preds)
    if "pre" in metrics_list:
        metrics["pre"] = recall_score(y_trues, y_preds, zero_division=0)
    if "rec" in metrics_list:
        metrics["rec"] = precision_score(y_trues, y_preds, zero_division=0)
    if "f1" in metrics_list:
        metrics["f1"] = f1_score(y_trues, y_preds, zero_division=0)

    return metrics

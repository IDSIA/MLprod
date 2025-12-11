from mlprod.database import crud, DataBase
from mlprod.worker.celery import worker
from mlprod.worker.models import Model, train_model, evaluate

from celery import Task
from datetime import datetime
from pathlib import Path

import os
import logging
import pandas as pd

LOGGER = logging.getLogger("mlprod.worker.tasks.training")

DEFAULT_MODEL_DIR = Path(".") / "models"


class TrainingTask(Task):
    """Abstraction of Celery's Task class."""

    abstract = True

    def __init__(self) -> None:
        """Initialize the TrainingTask."""
        super().__init__()

    def __call__(self, *args, **kwargs):
        """Call the run method of the task."""
        return self.run(*args, **kwargs)


@worker.task(
    ignore_results=True,
    bind=True,
    base=TrainingTask,
)
def training(self: TrainingTask):
    """Execute model training."""
    with DataBase().session() as session:
        try:
            # the task_id will also be the model id
            task_id = str(self.request.id)

            # folder to store models need to be created before saving
            folder_name = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
            path = DEFAULT_MODEL_DIR / f"model.{folder_name}"

            crud.update_model(session, task_id, "SETUP", path)

            LOGGER.info(f"Beginning model {task_id} training to path {path}")

            os.makedirs(path)

            # start training
            crud.update_model(session, task_id, "TRAINING", path)

            # load existing model for metadata
            db_model_old = crud.get_active_model(session)
            model_old = Model(db_model_old.path)

            tr_size, ts_size = model_old.metadata["n_records"], 1000
            features = model_old.metadata["features"]

            cols = features + ["label"]

            # create dataset
            df: pd.DataFrame = crud.create_dataset(session, task_id, tr_size + ts_size)[
                cols
            ]

            df_train = df[:tr_size]
            df_test = df[tr_size:]

            # list of metrics to check (same as declared in tables script)
            metrics_list = ["acc", "pre", "rec", "f1", "auc"]

            metrics_tr = train_model(df_train, path=path, metrics_list=metrics_list)

            # reload new trained model
            model_new = Model(path)

            # evaluate old and new model on test dataset
            X = df_test.drop("label", axis=1).values
            Y = df_test["label"].values.reshape(-1, 1)  # type: ignore

            y_preds_old = model_old(X)
            y_preds_new = model_new(X)

            model_old_metrics_ts = evaluate(Y, y_preds_old, metrics_list)
            model_new_metrics_ts = evaluate(Y, y_preds_new, metrics_list)

            acc_new = model_old_metrics_ts["auc"]
            acc_old = model_new_metrics_ts["auc"]

            # check if the new model is better than the old one
            if acc_new > acc_old:
                LOGGER.info(
                    f"Training {task_id} has better ROC AUC ({acc_new:.4}) than old model ({acc_old:.4})"
                )
                use_percentage = 1.0
            else:
                LOGGER.info(
                    f"Training {task_id} has worst ROC AUC ({acc_new:.4}) than old model ({acc_old:.4})"
                )
                use_percentage = 0.0

            # update entry in model table
            model_new_metrics = {
                "train": metrics_tr,
                "test": model_new_metrics_ts,
            }

            crud.update_model(
                session,
                task_id,
                "SUCCESS",
                metrics=model_new_metrics,
                use_percentage=use_percentage,
            )

            LOGGER.info(f"Training model {task_id} completed")

        except Exception as e:
            LOGGER.error(f"Training model {task_id} failed")
            crud.update_model(session, task_id, "FAILED")
            raise e

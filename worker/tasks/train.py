from celery import Task

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import Model, train_model, evaluate

from datetime import datetime

import os
import logging


class TrainingTask(Task):
    """"""

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


@worker.task(
    ignore_results=True,
    bind=True,
    base=TrainingTask
)
def training(self):
    try:
        task_id = str(self.request.id)
        db = SessionLocal()

        folder_name = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
        path = os.path.join('.', 'models', f'model.{folder_name}')
        logging.info(f'Beginning model {task_id} training to path {path}')

        os.makedirs(path)

        crud.update_model(db, task_id, 'TRAINING', path) 

        # load existing model
        db_model_old = crud.get_best_model(db)
        model_old = Model(db_model_old.path)

        tr_size, ts_size = model_old.metadata['n_records'], 1000
        features = model_old.metadata['features']

        cols = features + ['label']

        df = crud.create_dataset(db, task_id, tr_size + ts_size)

        df = df[cols]

        df_train = df[:tr_size]
        df_test = df[tr_size:]

        metrics_list = ['acc', 'pre', 'rec', 'f1', 'auc']
        metrics_tr = train_model(df_train, path=path, metrics_list=metrics_list)

        # reload trained model
        model_new = Model(path)

        # test with old and new model
        X = df_test.drop('label', axis=1).values
        Y = df_test['label'].values.reshape(-1, 1)

        y_preds_old = model_old(X)
        y_preds_new = model_new(X)

        model_old_metrics_ts = evaluate(Y, y_preds_old, metrics_list)
        model_new_metrics_ts = evaluate(Y, y_preds_new, metrics_list)

        acc_new = model_old_metrics_ts['auc']
        acc_old = model_new_metrics_ts['auc']

        if acc_new > acc_old:
            logging.info(f'Training {task_id} has better ROC AUC ({acc_new:.4}) than old model ({acc_old:.4})')
            use_percentage = 1.0
        else:
            logging.info(f'Training {task_id} has worst ROC AUC ({acc_new:.4}) than old model ({acc_old:.4})')
            use_percentage = 0.0

        # update entry in model table
        model_new_metrics = {
            'train': metrics_tr,
            'test': model_new_metrics_ts,
        }

        crud.update_model(db, task_id, 'SUCCESS', metrics=model_new_metrics, use_percentage=use_percentage)

        logging.info(f'Training model {task_id} completed')

    except Exception as e:
        logging.error(f'Training model {task_id} failed')
        crud.update_model(db, task_id, 'FAILED')
        raise e

    finally:
        db.close()

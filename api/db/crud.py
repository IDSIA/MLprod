from sqlalchemy.orm import Session

from datetime import datetime

from . import models, schemas

   
def get_prediction(db: Session, task_id: str) -> models.Prediction:
    return db.query(models.Prediction).filter(models.Prediction.task_id == task_id).first()


def create_prediction(db: Session, pred: schemas.PredictionCreate) -> models.Prediction:
    db_pred = models.Prediction(
        task_id = pred.task_id,
        x = pred.x,
        status = pred.status,
        time_post = datetime.now(),
    )
    db.add(db_pred)
    db.commit()
    db.refresh()
    return db_pred


def update_prediction(db: Session, pred: schemas.Prediction):
    db_pred = get_prediction(pred.task_id)

    db_pred.update({
        db_pred.time_get: pred.time_get,
        db_pred.status: pred.status
    })
    db.commit()
    db.refresh()
    return db_pred

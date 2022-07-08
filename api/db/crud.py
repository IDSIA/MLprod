from sqlalchemy.orm import Session

from . import schemas, tables

   
def get_prediction(db: Session, task_id: str) -> tables.Prediction:
    """Extract from the database the first Celery's task that match the given task_id.

    :param db:
      Session with the connection to the database.
    :param task_id:
      The id associated to the task.
    """
    return db.query(tables.Prediction).filter(tables.Prediction.task_id == task_id).first()


def create_prediction(db: Session, pred: schemas.PredictionCreate) -> tables.Prediction:
    """Insert a new prediction in the database.
    
    :param db:
      Session with the connection to the database.
    :param pred:
      Prediction object with the required fields
    """
    db_pred = tables.Prediction(
        task_id = pred.task_id,
        x = pred.x,
        status = pred.status,
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred


def update_prediction(db: Session, pred: schemas.Prediction) -> tables.Prediction:
    """Upadte a new prediction with the results.

    :param db:
      Session with the connection to the database.
    :param pred:
      Prediction object with the required fields
    """
    db_pred = get_prediction(db, pred.task_id)

    db_pred.time_get = pred.time_get
    db_pred.status = pred.status
    db.commit()
    db.refresh(db_pred)
    return db_pred


def create_event(db: Session, event: str) -> tables.Event:
    """Insert a new event into thte database.
    
    :param db:
      Session with the connection to the database.
    :param event:
      Event to be registered in the database. Technically, it is a string field,
      avoid typos and put single words.
    """
    db_event = tables.Event(
        event=event
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

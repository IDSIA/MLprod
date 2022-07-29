import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

from datetime import datetime
from .tables import Location, Inference, Event, Result, User


def create_user_data(db: Session, user_data: dict) -> User:
    """Store the data from a user in the database."""
    data = dict() | user_data
    ages = np.array(data['people_age'], dtype='float')

    data['age_avg'] = ages.mean()
    data['age_std'] = ages.std()
    data['age_min'] = ages.min()
    data['age_max'] = ages.max()

    del data['people_age']

    db_user = User(**data)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_inference(db: Session, task_id: str, status: str) -> Inference:
    """Insert a new inference in the database.
    
    :param db:
      Session with the connection to the database.
    :param task_id:
      Generated id for this task.
    :param status:
      Inirial status for this task.
    """
    db_pred = Inference(task_id=task_id, status=status)
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred


def get_inference(db: Session, task_id: str) -> Inference:
    """Extract from the database the first Celery's task that match the given task_id.

    :param db:
      Session with the connection to the database.
    :param task_id:
      The id associated to the task.
    """
    return db.query(Inference).filter(Inference.task_id == task_id).first()


def update_inference(db: Session, task_id: str, status: str) -> Inference:
    """Upadte an existing inference with the results.

    :param db:
      Session with the connection to the database.
    :param pred:
      Task id to update.
    :param status:
      New status to assign to the given task id.
    """
    db_pred = get_inference(db, task_id)
    db_pred.time_get = datetime.now()
    db_pred.status = status

    db.commit()
    db.refresh(db_pred)
    return db_pred


def create_results(db: Session, df: pd.DataFrame) -> list[Result]:
    db_results = []
    for _   , row in df.iterrows():
        result = Result(
            user_id=row['user_id'],
            location_id=row['location_id'],
            score=row['score'],
        )

        db.add(result)
        db_results.append(result)

    db.commit()
    for db_result in db_results:
        db.refresh(db_result)
    return db_results

def create_event(db: Session, event: str) -> Event:
    """Insert a new event into thte database.
    
    :param db:
      Session with the connection to the database.
    :param event:
      Event to be registered in the database. Technically, it is a string field,
      avoid typos and put single words.
    """
    db_event = Event(
        event=event
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event


def get_location(db: Session, id: int) -> Location:
    return db.query(Location).filter(Location.id == id).first()


def get_locations(db: Session) -> list[Location]:
    #TODO: add limit to this query?
    return db.query(Location).all()


def count_locations(db: Session) -> int:
    return db.query(Location).count()


def get_user(db: Session, id: int) -> User:
  return db.query(User).filter(User.id == id).first()


def get_users(db: Session) -> User:
  return db.query(User).all()


def count_users(db: Session) -> int:
    return db.query(User).count()

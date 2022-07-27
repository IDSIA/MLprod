from sqlalchemy.orm import Session
import numpy as np

from . import schemas
from .tables import Location, Prediction, Event, User

from ..requests import UserData

def create_userData(db: Session, user_data: UserData) -> User:
    data = dict()
    data.update(**user_data.__dict__)
    ages = np.array(user_data.people_age, dtype='float')

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


def get_prediction(db: Session, task_id: str) -> Prediction:
    """Extract from the database the first Celery's task that match the given task_id.

    :param db:
      Session with the connection to the database.
    :param task_id:
      The id associated to the task.
    """
    return db.query(Prediction).filter(Prediction.task_id == task_id).first()


def create_prediction(db: Session, pred: schemas.PredictionCreate) -> Prediction:
    """Insert a new prediction in the database.
    
    :param db:
      Session with the connection to the database.
    :param pred:
      Prediction object with the required fields
    """
    db_pred = Prediction(
        task_id = pred.task_id,
        status = pred.status,
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred


def update_prediction(db: Session, pred: schemas.Prediction) -> Prediction:
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
    #TODO: add limit to this query
    return db.query(Location).all()


def count_locations(db: Session) -> int:
    return db.query(Location).count()


def count_users(db: Session) -> int:
    return db.query(User).count()

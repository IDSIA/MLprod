import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

from datetime import datetime
from .tables import Location, Inference, Event, Result, User, Model


def create_user_data(db: Session, user_data: dict) -> User:
    """Store the data from a user in the database.
    
    :param db:
        Session with the connection to the database.
    :param user_data:
        Content to be saved to the database.
    """
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
    """Update an existing inference with the results.

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
    """Creates a new result from a Pandas' DataFrame. This DataFrame is the 
    Output of an inference call of our ML model.

    :param db:
        Session with the connection to the database.
    :param df:
        A dataframe with the columns 'user_id', 'location_id', 'score', and
        'task_id'.
    """
    db_results = []
    for _, row in df.iterrows():
        result = Result(
            user_id=row['user_id'],
            location_id=row['location_id'],
            score=row['score'],
            task_id=row['task_id'],
            label=0,
        )

        db.add(result)
        db_results.append(result)

    db.commit()
    
    for db_result in db_results:
        db.refresh(db_result)

    return db_results


def get_results_locations(db: Session, task_id: str, limit: int=10) -> list[dict]:
    """Get all the scored results based on the """
    db_results = (
        db.query(Result)
        .join(Location, Result.location_id == Location.id)
        .filter(Result.task_id == task_id)
        .order_by(Result.score.desc())
        .limit(limit)
        .all()
    )

    results = []
    for db_result in db_results:
        results.append({
            'score': db_result.score,
            'location_id': db_result.location.id,
            'children': db_result.location.children,
            'breakfast': db_result.location.breakfast,
            'lunch': db_result.location.lunch,
            'dinner': db_result.location.dinner,
            'price': db_result.location.price,
            'pool': db_result.location.pool,
            'spa': db_result.location.spa,
            'animals': db_result.location.animals,
            'lake': db_result.location.lake,
            'mountain': db_result.location.mountain,
            'sport': db_result.location.sport,
            'family_rating': db_result.location.family_rating,
            'outdoor_rating': db_result.location.outdoor_rating,
            'food_rating': db_result.location.food_rating,
            'leisure_rating': db_result.location.leisure_rating,
            'service_rating': db_result.location.service_rating,
            'user_score': db_result.location.user_score,
        })

    return results


def get_results(db: Session, task_id: int) -> list[Result]:
    return db.query(Result).filter(Result.task_id == task_id).all()


def get_result(db: Session, result_id: int) -> Result:
    return db.query(Result).filter(Result.id == result_id).first()


def update_result_label(db: Session, task_id: str, location_id: int) -> Result:
    """Updates the result identified by task_id and location_id by assigning 
    the label 1 (default is 0).
    
    :param db:
        Session with the connection to the database.
    :param task_id:
        Id of the task to update.
    :param location_id:
        Id of the location to update.
    """
    db_result = db.query(Result).filter(Result.task_id == task_id).filter(Result.location_id == location_id).first()

    if db_result is None:
        return None
    
    db_result.label = 1
    db.commit()
    db.refresh(db_result)

    return db_result


def create_model(db: Session, path: str, metrics: dict[str, float], use_percentage: float=.0) -> Model:
    db_model = Model(path=path, use_percentage=use_percentage, **metrics)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def get_best_model(db: Session) -> Model:
    # TODO: this should return a list of models, then who call it choose what to load
    return db.query(Model).filter(Model.use_percentage > 0).first()


def count_models(db: Session) -> int:
    return db.query(Model).count()


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


def get_locations(db: Session, limit: int=0) -> list[Location]:
    if limit > 0:
        return db.query(Location).limit(limit).all()

    return db.query(Location).all()


def count_locations(db: Session) -> int:
    return db.query(Location).count()


def get_user(db: Session, id: int) -> User:
    return db.query(User).filter(User.id == id).first()


def get_users(db: Session) -> User:
    return db.query(User).all()


def count_users(db: Session) -> int:
    return db.query(User).count()

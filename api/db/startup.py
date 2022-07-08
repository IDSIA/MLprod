from api.db.tables import Prediction, Event, Request, POI, Dataset
from sqlalchemy.orm import Session


def init_content(db: Session):
    """"
    Initialize all the tables in the database, if they do not exists.

    :param db:
      Session with the connection to the database."""
    engine = db.get_bind()
    Prediction.__table__.create(bind=engine, checkfirst=True)
    
    Event.__table__.create(bind=engine, checkfirst=True)

    Request.__table__.create(bind=engine, checkfirst=True)
    POI.__table__.create(bind=engine, checkfirst=True)
    Dataset.__table__.create(bind=engine, checkfirst=True)

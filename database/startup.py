from .tables import User, Location, Dataset, Inference, Result, Event
from .crud import count_locations
from sqlalchemy.orm import Session

import pandas as pd


def init_content(db: Session):
    """"
    Initialize all the tables in the database, if they do not exists.

    :param db:
      Session with the connection to the database."""
    engine = db.get_bind()
    
    User.__table__.create(bind=engine, checkfirst=True)
    Location.__table__.create(bind=engine, checkfirst=True)
    Dataset.__table__.create(bind=engine, checkfirst=True)

    Inference.__table__.create(bind=engine, checkfirst=True)
    Result.__table__.create(bind=engine, checkfirst=True)

    Event.__table__.create(bind=engine, checkfirst=True)

    n_locations = count_locations(db)

    if n_locations == 0:
        df = pd.read_csv('./dataset/dataset_locations.tsv', sep='\t')

        db.bulk_insert_mappings(Location, df.to_dict(orient='records'))
        db.commit()
        db.close()

from api.db.models import Prediction
from sqlalchemy.orm import Session


def init_content(db: Session):
    engine = db.get_bind()
    Prediction.__table__.create(bind=engine, checkfirst=True)

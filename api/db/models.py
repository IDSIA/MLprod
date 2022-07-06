from sqlalchemy import Column, String, Float, DateTime

from api.db.database import Base


class Prediction(Base):
    __tablename__ = 'predictions'
    task_id  = Column(String, primary_key=True, index=True)
    time_post = Column(DateTime(timezone=True), default=None)
    time_get = Column(DateTime(timezone=True), default=None)
    x = Column(Float, default=None)
    y = Column(Float, default=None)
    status = Column(String, default='')

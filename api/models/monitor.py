from pydantic import DateTimeError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, DateTime

from sqlalchemy.sql import func

SQLBase = declarative_base()

class Prediction(SQLBase):
    __tablename__ = 'prediction'
    task_id  = Column(String, primary_key=True, index=True)
    time_post = Column(DateTime(timezone=True), server_default=func.now())
    time_get = Column(DateTime(timezone=True), server_default=None)
    x = Column(Float, default=None)
    y = Column(Float, default=None)
    status = Column(String)

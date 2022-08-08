from sqlalchemy import Column, ForeignKey, String, Float, DateTime, Integer, Boolean, Date
from sqlalchemy.sql.functions import now
from sqlalchemy.orm import relationship

from .database import Base


# ---- Dataset tables ----

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    creation_time = Column(DateTime(timezone=True), server_default=now())
    people_num = Column(Integer, nullable=False)
    children = Column(Integer, nullable=False)
    age_avg = Column(Float, nullable=False)
    age_std = Column(Float, nullable=False)
    age_min = Column(Float, nullable=False)
    age_max = Column(Float, nullable=False)
    budget = Column(Integer, nullable=False)
    nights = Column(Integer, nullable=False)
    time_arrival = Column(Date, nullable=True)
    pool = Column(Boolean, default=False)
    spa = Column(Boolean, default=False)
    pet_friendly = Column(Boolean, default=False)
    lake = Column(Boolean, default=False)
    mountain = Column(Boolean, default=False)
    sport = Column(Boolean, default=False)


class Location(Base):
    __tablename__ = 'locations'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    creation_time = Column(DateTime(timezone=True), server_default=now())
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    children = Column(Boolean, nullable=False)
    breakfast = Column(Boolean, nullable=False)
    lunch = Column(Boolean, nullable=False)
    dinner = Column(Boolean, nullable=False)
    price = Column(Float, nullable=False)
    pool = Column(Boolean, default=False)
    spa = Column(Boolean, default=False)
    animals = Column(Boolean, default=False)
    lake = Column(Boolean, default=False)
    mountain = Column(Boolean, default=False)
    sport = Column(Boolean, default=False)
    family_rating = Column(Float, nullable=False)
    outdoor_rating = Column(Float, nullable=False)
    food_rating = Column(Float, nullable=False)
    leisure_rating = Column(Float, nullable=False)
    service_rating = Column(Float, nullable=False)
    user_score = Column(Float, nullable=False)


# ---- Inference tables ----

class Inference(Base):
    """Table used to store the inference requests from the users, and the results."""
    __tablename__ = 'inferences'
    task_id = Column(String, primary_key=True, index=True)
    time_creation = Column(DateTime(timezone=True), server_default=now())
    time_get = Column(DateTime(timezone=True), default=None)
    time_update = Column(DateTime(timezone=True), default=None)
    user_id = Column(Integer, ForeignKey('users.id'))
    status = Column(String, default='')

    user = relationship('User')


class Result(Base):
    """Table used to store the inference results from the ML model."""
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    task_id = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    location_id = Column(Integer, ForeignKey('locations.id'))
    score = Column(Float, nullable=False)
    label = Column(Integer, default=0)
    shown = Column(Boolean, default=False)

    user = relationship('User')
    location = relationship('Location')


# ---- Monitoring and metrics tables ----

class Event(Base):
    """Table used to store the events that can be generated by the application.
    This is very similar to a log file."""
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String, default='')
    time_event = Column(DateTime(timezone=True), server_default=now())


# ---- Retraining tables ----

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey('results.id'), primary_key=True)
    time_creation = Column(DateTime(timezone=True), server_default=now())

    result = relationship('Result')


class Model(Base):
    """Table used to store information regarding models generated by teh application."""
    __tablename__ = 'models'
    task_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    time_creation = Column(DateTime(timezone=True), server_default=now())
    status = Column(String, default='')
    path = Column(String)
    use_percentage = Column(Float, default=0.0)
    train_acc = Column(Float, default=0.0)
    train_auc = Column(Float, default=0.0)
    train_pre = Column(Float, default=0.0)
    train_rec = Column(Float, default=0.0)
    train_f1 = Column(Float, default=0.0)
    test_acc = Column(Float, default=0.0)
    test_auc = Column(Float, default=0.0)
    test_pre = Column(Float, default=0.0)
    test_rec = Column(Float, default=0.0)
    test_f1 = Column(Float, default=0.0)
    # TODO: add used dataset

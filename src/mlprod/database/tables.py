from pathlib import Path
from sqlalchemy import TypeDecorator, ForeignKey, String, DateTime, Date
from sqlalchemy.sql.functions import now
from sqlalchemy.orm import relationship, mapped_column, Mapped, DeclarativeBase

from datetime import datetime, date


# ---- Decorators ----


class PathType(TypeDecorator):
    """Decorator to enable the use of PathLike objects."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Path | str | None, dialect) -> str | None:
        """Converts the path to string."""
        if value is None:
            return None
        # Accept both Path objects and raw strings
        return str(value)

    def process_result_value(self, value: str | None, dialect) -> Path | None:
        """Converts a string to a Path object."""
        if value is None:
            return None
        return Path(value)


# ---- Dataset tables ----
class Base(DeclarativeBase):
    """Base class for all the database tables."""

    pass


class User(Base):
    """Table used to store all the data received with the user's requests."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, index=True
    )
    creation_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )
    name: Mapped[str] = mapped_column(nullable=False)
    people_num: Mapped[int] = mapped_column(nullable=False)
    children_num: Mapped[int] = mapped_column(nullable=False)
    age_avg: Mapped[float] = mapped_column(nullable=False)
    age_std: Mapped[float] = mapped_column(nullable=False)
    age_min: Mapped[float] = mapped_column(nullable=False)
    age_max: Mapped[float] = mapped_column(nullable=False)
    budget: Mapped[int] = mapped_column(nullable=False)
    nights: Mapped[int] = mapped_column(nullable=False)
    time_arrival: Mapped[date] = mapped_column(Date, nullable=True)
    pool: Mapped[bool] = mapped_column(default=False)
    spa: Mapped[bool] = mapped_column(default=False)
    pet_friendly: Mapped[bool] = mapped_column(default=False)
    lake: Mapped[bool] = mapped_column(default=False)
    mountain: Mapped[bool] = mapped_column(default=False)
    sport: Mapped[bool] = mapped_column(default=False)


class Location(Base):
    """Table used to store all the locations."""

    __tablename__ = "locations"

    location_id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, index=True
    )
    creation_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )
    lat: Mapped[float] = mapped_column(nullable=True)
    lon: Mapped[float] = mapped_column(nullable=True)
    children: Mapped[bool] = mapped_column(nullable=False)
    breakfast: Mapped[bool] = mapped_column(nullable=False)
    lunch: Mapped[bool] = mapped_column(nullable=False)
    dinner: Mapped[bool] = mapped_column(nullable=False)
    price: Mapped[float] = mapped_column(nullable=False)
    has_pool: Mapped[bool] = mapped_column(default=False)
    has_spa: Mapped[bool] = mapped_column(default=False)
    animals: Mapped[bool] = mapped_column(default=False)
    near_lake: Mapped[bool] = mapped_column(default=False)
    near_mountains: Mapped[bool] = mapped_column(default=False)
    has_sport: Mapped[bool] = mapped_column(default=False)
    family_rating: Mapped[float] = mapped_column(nullable=False)
    outdoor_rating: Mapped[float] = mapped_column(nullable=False)
    food_rating: Mapped[float] = mapped_column(nullable=False)
    leisure_rating: Mapped[float] = mapped_column(nullable=False)
    service_rating: Mapped[float] = mapped_column(nullable=False)
    user_score: Mapped[float] = mapped_column(nullable=False)


# ---- Inference tables ----


class Inference(Base):
    """Table used to store the inference requests from the users, and the results."""

    __tablename__ = "inferences"

    task_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    time_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )
    time_get: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=None, nullable=True
    )
    time_update: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=None, nullable=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    status: Mapped[str] = mapped_column(default="")

    user = relationship("User")


class Result(Base):
    """Table used to store the inference results from the ML model."""

    __tablename__ = "results"

    result_id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True, index=True
    )
    task_id: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.location_id"))
    score: Mapped[float] = mapped_column(nullable=False)
    label: Mapped[int] = mapped_column(default=0)
    shown: Mapped[bool] = mapped_column(default=False)

    user = relationship("User")
    location = relationship("Location")


# ---- Monitoring and metrics tables ----


class Event(Base):
    """Table used to store the events that can be generated by the application.

    This is very similar to a log file.
    """

    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    event: Mapped[str] = mapped_column(default="")
    time_event: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )


# ---- Retraining tables ----


class Dataset(Base):
    """Table used to store information on dataset used during the training of a model.

    Use the filed `task_id` to find the used model.
    """

    __tablename__ = "datasets"

    task_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    result_id: Mapped[int] = mapped_column(
        ForeignKey("results.result_id"), primary_key=True
    )
    time_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )

    result = relationship("Result")


class Model(Base):
    """Table used to store information regarding models generated by the training task.

    Use the field `task_id` to find the used dataset.
    """

    __tablename__ = "models"

    task_id: Mapped[str] = mapped_column(primary_key=True, index=True)
    time_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=now()
    )
    status: Mapped[str] = mapped_column(default="")
    path: Mapped[Path] = mapped_column(PathType)
    use_percentage: Mapped[float] = mapped_column(default=0.0)
    train_acc: Mapped[float] = mapped_column(default=0.0)
    train_auc: Mapped[float] = mapped_column(default=0.0)
    train_pre: Mapped[float] = mapped_column(default=0.0)
    train_rec: Mapped[float] = mapped_column(default=0.0)
    train_f1: Mapped[float] = mapped_column(default=0.0)
    test_acc: Mapped[float] = mapped_column(default=0.0)
    test_auc: Mapped[float] = mapped_column(default=0.0)
    test_pre: Mapped[float] = mapped_column(default=0.0)
    test_rec: Mapped[float] = mapped_column(default=0.0)
    test_f1: Mapped[float] = mapped_column(default=0.0)

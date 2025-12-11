from __future__ import annotations
from typing import Any, Generator

from sqlalchemy.engine import Engine, create_engine, URL
from sqlalchemy.orm import Session, sessionmaker

import logging
import os

LOGGER = logging.getLogger("mlprod.database")

DATABASE_URL = os.environ.get("DATABASE_URL", "")


class DataBase:
    """Singleton class to manage the connection to the database."""

    def __init__(self) -> None:
        """Initialize the database connection parameters."""
        self.database_url: URL | str | None
        self.engine: Engine
        self.session_factory: Any
        self.sync_session: Any

    def __new__(cls) -> DataBase:
        """Create a singleton instance of the DataBase class."""
        if not hasattr(cls, "instance"):
            LOGGER.debug("database singleton creation")
            cls.instance = super(DataBase, cls).__new__(cls)

            cls.instance.database_url = DATABASE_URL

            if cls.instance.database_url is None:
                raise ValueError("Connection to database is not set!")

            cls.instance.engine = create_engine(cls.instance.database_url)
            cls.instance.sync_session = sessionmaker(
                bind=cls.instance.engine,
                class_=Session,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            LOGGER.info("dataBase connection established")

        return cls.instance

    def session(self) -> Session:
        """Create a new session to interact with the database."""
        return self.sync_session()


def get_session() -> Generator[Session, None, None]:
    """This is a generator for obtain the session to the database through SQLAlchemy."""
    db = DataBase()
    with db.session() as session:
        yield session

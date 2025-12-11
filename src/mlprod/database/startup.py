from .crud import count_locations, count_models, create_model
from .database import DataBase
from .tables import Base
from .tables import Location

from pathlib import Path

import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("mlprod.database.startup")


def init_content() -> None:
    """Initialize all the tables in the database, if they do not exists."""
    LOGGER.info("server startup procedure started")

    try:
        db = DataBase()

        with db.engine.begin() as conn:
            LOGGER.info("database creation started")
            Base.metadata.create_all(conn, checkfirst=True)
            LOGGER.info("database creation completed")

        with db.session() as session:
            n_locations = count_locations(session)

            if n_locations == 0:
                LOGGER.info(
                    "no locations found in database, "
                    "populating from dataset_locations.tsv file"
                )

                df = pd.read_csv("./data/dataset_locations.tsv", sep="\t")
                df["id"] = np.arange(df.shape[0])

                session.bulk_insert_mappings(Location, df.to_dict(orient="records"))  # type: ignore
                session.commit()

            n_models = count_models(session)

            if n_models == 0:
                LOGGER.info("no models found in database, adding baseline model")

                create_model(
                    session,
                    "baseline_model",
                    "SUCCESS",
                    path=Path("/") / "app" / "models" / "original",
                    use_percentage=1.0,
                )

            session.close()

    except Exception as e:
        LOGGER.error("Error during database initialization")
        LOGGER.exception(e)

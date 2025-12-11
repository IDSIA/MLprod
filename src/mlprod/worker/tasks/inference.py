from celery import Task
from pathlib import Path

from mlprod.database import DataBase, crud
from mlprod.worker.celery import worker
from mlprod.worker.models import Model

import pandas as pd
import logging


LOGGER = logging.getLogger("mlprod.worker.tasks.inference")


class InferenceTask(Task):
    """Abstraction of Celery's Task class."""

    abstract = True

    def __init__(self) -> None:
        """Initialize the InferenceTask."""
        super().__init__()

        self.model: Model | None = None
        self.path: Path | None = None

    def __call__(self, *args, **kwargs) -> None:
        """Call the run method of the task."""
        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=InferenceTask,
)
def inference(self: InferenceTask, user_id: int) -> None:
    """Execute inference for the given user_id.

    :param user_id:
        ID of the new user
    """
    with DataBase().session() as session:
        # check if there is a new model to use
        db_model = crud.get_active_model(session)

        if self.model is None or self.path != db_model.path:
            LOGGER.info(f"Reloading model from path {db_model.path}")
            self.path = db_model.path
            self.model = Model(self.path)

        # get data to process
        user = crud.get_user(session, user_id)
        locs = crud.get_locations(session)

        locs_id = [loc.location_id for loc in locs]

        df = pd.DataFrame(
            [user.__dict__ | loc.__dict__ for loc in locs],
            columns=self.model.metadata["features"],
        )

        # apply model to data
        score = self.model(df.values)

        df["score"] = score
        df["user_id"] = user_id
        df["location_id"] = locs_id
        df["task_id"] = str(self.request.id)

        # save task id, user_id, and scores to database
        crud.create_results(session, df)

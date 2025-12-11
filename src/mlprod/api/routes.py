from contextlib import asynccontextmanager
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from mlprod.api.middleware.metrics import PrometheusMiddleware, metrics_route
from mlprod.api import requests
from mlprod.database import crud, init_content, get_session, DataBase
from mlprod.logs import setup_logs
from mlprod.worker.tasks.inference import inference
from mlprod.worker.tasks.train import training
from mlprod import __version__

import logging

setup_logs()

LOGGER = logging.getLogger("mlprod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    await startup()
    yield
    await shutdown()


def init_api() -> FastAPI:
    """Initialize the FastAPI app with routes and middleware."""
    api = FastAPI(
        title="MLprod API",
        version=__version__,
        lifespan=lifespan,
    )

    api.add_middleware(PrometheusMiddleware)
    api.add_route("/metrics", metrics_route)
    return api


api = init_api()


async def startup() -> None:
    """Initialize the database and populate it with some data."""
    init_content()


async def shutdown() -> None:
    """Dispose the database engine on shutdown."""
    LOGGER.info("server shutdown procedure started")
    inst = DataBase()
    if inst.engine:
        inst.engine.dispose()


@api.get("/")
def root():
    """This is the endpoint for the home page."""
    return "Hi! 😀"


@api.post("/inference/start")
async def schedule_inference(
    user_data: requests.UserData, db: Session = Depends(get_session)
):
    """This is the endpoint used for schedule an inference."""
    LOGGER.debug(f"Scheduling inference for user data: {user_data}")

    crud.create_event(db, "inference_start")
    ud = crud.create_user_data(db, user_data.model_dump())
    task: AsyncResult = inference.delay(ud.user_id)

    if task.task_id is None:
        LOGGER.error("Invalid task id returned from inference task")
        raise HTTPException(500, "Invalid task id")

    db_inf = crud.create_inference(db, task.task_id, task.status, ud.user_id)
    status = requests.TaskStatus(
        task_id=db_inf.task_id, status=db_inf.status, type="inference"
    )

    LOGGER.debug(f"Inference scheduled with status: {status}")

    return status


@api.get("/inference/status/{task_id}", response_model=requests.TaskStatus)
async def get_inference_status(task_id: str, db: Session = Depends(get_session)):
    """This is the endpoint to get the results of an inference."""
    crud.create_event(db, "status")

    task = AsyncResult(task_id)

    db_inf = crud.get_inference(db, task_id=task_id)

    if db_inf is None:
        LOGGER.error(f"Inference task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    db_inf.status = task.status
    db_inf = crud.update_inference(db, db_inf.task_id, db_inf.status)

    if task.failed():
        LOGGER.error(f"Inference task failed: {task_id}")
        raise HTTPException(status_code=500, detail="Task failed")

    if not task.ready():
        return requests.TaskStatus(
            task_id=db_inf.task_id,
            status=db_inf.status,
            type="inference",
        )

    return requests.TaskStatus(
        task_id=db_inf.task_id,
        status=db_inf.status,
        type="inference",
    )


@api.get("/inference/results/{task_id}")
async def get_inference_results(
    task_id: str, limit: int = 10, db: Session = Depends(get_session)
):
    """This is the endpoint to get the results with scores after the inference.

    Note: check the status of the task with the '/inference/status' endpoint.
    """
    crud.create_event(db, "results")

    locations = crud.get_results_locations(db, task_id, limit)

    crud.mark_locations_as_shown(db, task_id, locations)

    return locations


@api.put("/inference/select/")
async def get_click(label: requests.LabelData, db: Session = Depends(get_session)):
    """This is the endpoint used to simulate a click on a choice.

    A click will be registered as a label on the data.
    """
    crud.create_event(db, "selection")

    if label.location_id == -1:
        crud.create_event(db, "bad_inference")
        return

    else:
        crud.create_event(db, "good_inference")
        db_result = crud.update_result_label(db, label.task_id, label.location_id)

        if db_result is None:
            LOGGER.error(
                "Result not found for task_id "
                f"{label.task_id} and location_id {label.location_id}"
            )
            return HTTPException(
                404, "Result not found: invalid task_id or location_id"
            )

        return db_result


@api.post("/train/start")
async def schedule_training(db: Session = Depends(get_session)):
    """This is the endpoint to start the training of a new model."""
    crud.create_event(db, "training")

    task: AsyncResult = training.delay()

    db_model = crud.create_model(db, task.task_id or "", task.status)
    return requests.TaskStatus(
        task_id=db_model.task_id, status=db_model.status, type="training"
    )


@api.get("/content/info")
async def get_content_info(db: Session = Depends(get_session)):
    """This is the endpoint to get some information about the content in the database."""
    n_locations = crud.count_locations(db)
    n_users = crud.count_users(db)

    return requests.ContentInfo(
        locations=n_locations,
        users=n_users,
    )


@api.get("/content/location/{location_id}")
async def get_content_location(location_id: int, db: Session = Depends(get_session)):
    """This is the endpoint to get a location by its ID."""
    return crud.get_location(db, location_id)


@api.get("/content/locations")
async def get_content_locations(db: Session = Depends(get_session)):
    """This is the endpoint to get all the locations."""
    return crud.get_locations(db)


@api.get("/content/user/{user_id}")
async def get_content_user(user_id: int, db: Session = Depends(get_session)):
    """This is the endpoint to get a user by its ID."""
    return crud.get_user(db, user_id)


@api.get("/content/users")
async def get_content_users(db: Session = Depends(get_session)):
    """This is the endpoint to get all the users."""
    return crud.get_users(db)


@api.get("/content/result/{result_id}")
async def get_content_result_byid(result_id: int, db: Session = Depends(get_session)):
    """This is the endpoint to get a result by its ID."""
    return crud.get_result(db, result_id)


@api.get("/content/results/{result_id}")
async def get_content_result(task_id: int, db: Session = Depends(get_session)):
    """This is the endpoint to get all results for a given task ID."""
    return crud.get_results(db, task_id)

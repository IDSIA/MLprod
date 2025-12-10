from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from mlprod.api.middleware.metrics import PrometheusMiddleware, metrics_route
from mlprod.api import requests

from mlprod.database.database import SessionLocal
from mlprod.database import crud, startup

from mlprod.worker.tasks.inference import inference
from mlprod.worker.tasks.train import training

import uvicorn


api = FastAPI()

api.add_middleware(PrometheusMiddleware)
api.add_route("/metrics", metrics_route)


def get_db():
    """This is a generator for obtain the session to the database through SQLAlchemy."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.on_event("startup")
async def populate_database():
    """All operations marked as ``on_event('startup')`` are executed when the APi are runned.

    In this case, we initialize the database and populate it with some data.
    """
    try:
        db = SessionLocal()
        startup.init_content(db)
    finally:
        db.close()


@api.get("/")
def root():
    """This is the endpoint for the home page."""
    return "Hi! 😀"


@api.post("/inference/start")
async def schedule_inference(
    user_data: requests.UserData, db: Session = Depends(get_db)
):
    """This is the endpoint used for schedule an inference."""
    crud.create_event(db, "inference_start")
    ud = crud.create_user_data(db, user_data.model_dump())
    task: AsyncResult = inference.delay(ud.user_id)

    if task.task_id is None:
        raise HTTPException(500, "Invalid task id")

    db_inf = crud.create_inference(db, task.task_id, task.status)
    return requests.TaskStatus(
        task_id=db_inf.task_id, status=db_inf.status, type="inference"
    )


@api.get("/inference/status/{task_id}", response_model=requests.TaskStatus)
async def get_inference_status(task_id: str, db: Session = Depends(get_db)):
    """This is the endpoint to get the results of an inference."""
    crud.create_event(db, "status")

    task = AsyncResult(task_id)
    task_id, status = task.task_id, task.status

    db_inf = crud.get_inference(db, task_id=task_id)

    if db_inf is None:
        raise HTTPException(status_code=404, detail="Task not found")

    db_inf.status = status
    db_inf = crud.update_inference(db, db_inf.task_id, db_inf.status)

    if task.failed():
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
    task_id: str, limit: int = 10, db: Session = Depends(get_db)
):
    """This is the endpoint to get the results with scores after the inference.

    Note: check the status of the task with the '/inference/status' endpoint.
    """
    crud.create_event(db, "results")

    locations = crud.get_results_locations(db, task_id, limit)

    crud.mark_locations_as_shown(db, task_id, locations)

    return locations


@api.put("/inference/select/")
async def get_click(label: requests.LabelData, db: Session = Depends(get_db)):
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
            return HTTPException(
                404, "Result not found: invalid task_id or location_id"
            )

        return db_result


@api.post("/train/start")
async def schedule_training(db: Session = Depends(get_db)):
    """This is the endpoint to start the training of a new model."""
    crud.create_event(db, "training")

    task: AsyncResult = training.delay()

    db_model = crud.create_model(db, task.task_id, task.status)
    return requests.TaskStatus(
        task_id=db_model.task_id, status=db_model.status, type="training"
    )


@api.get("/content/info")
async def get_content_info(db: Session = Depends(get_db)):
    n_locations = crud.count_locations(db)
    n_users = crud.count_users(db)

    return requests.ContentInfo(
        locations=n_locations,
        users=n_users,
    )


@api.get("/content/location/{location_id}")
async def get_content_location(location_id: int, db: Session = Depends(get_db)):
    return crud.get_location(db, location_id)


@api.get("/content/locations")
async def get_content_locations(db: Session = Depends(get_db)):
    return crud.get_locations(db)


@api.get("/content/user/{user_id}")
async def get_content_user(user_id: int, db: Session = Depends(get_db)):
    return crud.get_user(db, user_id)


@api.get("/content/users")
async def get_content_users(db: Session = Depends(get_db)):
    return crud.get_users(db)


@api.get("/content/result/{result_id}")
async def get_content_result_byid(result_id: int, db: Session = Depends(get_db)):
    return crud.get_result(db, result_id)


@api.get("/content/results/{result_id}")
async def get_content_result(task_id: int, db: Session = Depends(get_db)):
    return crud.get_results(db, task_id)


if __name__ == "__main__":
    uvicorn.run(api)

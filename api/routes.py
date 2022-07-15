from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

import uvicorn

from api.middleware.metrics import PrometheusMiddleware, metrics_route
from api.db.database import SessionLocal
from api.db import crud, schemas, startup
from api import requests

from worker.pred import predict


api = FastAPI()

api.add_middleware(PrometheusMiddleware)
api.add_route('/metrics', metrics_route)


def get_db():
    """This is a generator for obtain the session to the database through SQLAlchemy."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.on_event('startup')
async def populate_database():
    """
    All operations marked as ``on_event('startup')`` are executed when the APi are runned.

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
    return 'Hi! 😀'


@api.post('/pred', response_model=requests.PredictionOutput)
async def schedule_prediction(input_x: requests.PredictionInput, db: Session = Depends(get_db)):
    """This is the endpoint used for schedule an inference."""
    crud.create_event('prediction')

    x = float(input_x.x)
    task: AsyncResult = predict.delay(x)
    
    prediction = schemas.PredictionCreate(
        task_id=task.task_id,
        x=x,
        status=task.status,
    )

    db_pred = crud.create_prediction(db, prediction)
    return requests.PredictionOutput(
        task_id=db_pred.task_id,
        status=db_pred.status,
        y=None
    )


@api.get('/result/{task_id}', response_model=requests.PredictionOutput)
async def get_results(task_id: str, db: Session = Depends(get_db)):
    """This si the endpoint to get the results of an inference."""
    crud.create_event('results')
    
    task = AsyncResult(task_id)
    task_id, status = task.task_id, task.status

    db_pred = crud.get_prediction(db, task_id=task_id)

    if db_pred is None:
        raise HTTPException(status_code=404, detail='Task not found')

    db_pred.status = status

    if task.failed():
        crud.update_prediction(db, db_pred)
        raise HTTPException(status_code=500, detail='Task failed')

    if not task.ready():
        return requests.PredictionOutput(
            task_id=db_pred.task_id,
            status=db_pred.task_id,
            y = None
        )

    db_pred.y = task.get()

    db_pred = crud.update_prediction(db, db_pred)

    return requests.PredictionOutput(
        task_id=db_pred.task_id,
        status=db_pred.task_id,
        y = db_pred.y
    )


@api.get('/post/{choice}')
async def get_click(choice: str, db: Session = Depends(get_db)):
    """This is the endpoint used to simulate a click on a choice.
    A click will be registered as a label on the data"""
    
    return HTTPException(501, 'Not implemented')


if __name__ == '__main__':
    uvicorn.run(api)

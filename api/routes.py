from worker.tasks import add, mul
from worker.pred import predict
from celery.result import AsyncResult

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi_sqlalchemy import DBSessionMiddleware, db 

from api.models.requests import Inputs, PredictionInput, TaskOutput, TaskResult
from api.models.monitor import Prediction as ModelPrediction
from api.models.schema import Prediction

import os


api = FastAPI()

api.add_middleware(DBSessionMiddleware, db_url=os.environ.get('DATABASE_URL'))


@api.get("/")
def root():
    return 'Hi! 😀'


@api.post('/add', response_model=TaskOutput, status_code=200)
async def schedule_add(model_input: Inputs):
    task = add.delay(model_input.x, model_input.y)
    return TaskOutput(
        task_id=str(task), 
        status=task.status
    )


@api.post('/mul', response_model=TaskOutput, status_code=200)
async def schedule_add(model_input: Inputs):
    task = mul.delay(model_input.x, model_input.y)
    return TaskOutput(
        task_id=str(task), 
        status=task.status
    )


@api.post('/pred', response_model=TaskOutput, status_code=200)
async def schedule_prediction(model_input: PredictionInput):
    task = predict.delay(model_input.x)
    task_id, status = str(task), task.status

    db_pred = ModelPrediction(task_id=task_id, x=model_input.x, y=None, status=status)
    db.session.add(db_pred)
    db.session.commit()

    return TaskOutput(
        task_id=task_id, 
        status=status
    )


@api.get('/result/{task_id}', response_model=TaskResult, status_code=200)
async def get_results(task_id):
    task = AsyncResult(task_id)
    task_id, status = str(task), task.status

    db_pred = db.session.query(ModelPrediction).filter(ModelPrediction.task_id == task_id)
    db_pred.status = status

    if task.failed():
        db.session.add(db_pred)
        db.session.commit()

        return JSONResponse(status_code=500, content=TaskOutput(
            task_id=str(task), 
            status=status
        ))

    if not task.ready():
        return JSONResponse(status_code=200, content=TaskOutput(
            task_id=str(task), 
            status=status
        ))
    
    y = task.get()

    db_pred.y = y
    db.session.add(db_pred)
    db.session.commit()

    return TaskResult(
        task_id=task_id, 
        status=status,
        y=y
    )

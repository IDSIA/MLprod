from worker.tasks import add, mul
from celery.result import AsyncResult

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from api.models import *

api = FastAPI()


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


@api.get('/result/{task_id}', response_model=TaskResult, status_code=200)
async def get_results(task_id):
    task = AsyncResult(task_id)

    if not task.ready():
        return JSONResponse(status_code=200, content=TaskOutput(
            task_id=str(task), 
            status=task.status
        ))

    if task.failed():
        return JSONResponse(status_code=500, content=TaskOutput(
            task_id=str(task), 
            status=task.status
        ))
    
    return TaskResult(
        task_id=str(task), 
        status=task.status,
        y=task.get()
    )

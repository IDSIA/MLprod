from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from api.db.database import SessionLocal
from api.db import crud, schemas, startup

from worker.pred import predict

from datetime import datetime


api = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.on_event('startup')
async def populate_database():
    try:
        db = SessionLocal()
        startup.init_content(db)
    finally:
        db.close()


@api.get("/")
def root():
    return 'Hi! 😀'


@api.post('/pred', response_model=schemas.Prediction)
async def schedule_prediction(input_x: float, db: Session = Depends(get_db)):
    task = predict.delay(input_x)
    
    prediction = schemas.PredictionCreate(
        task_id=str(task),
        x=schemas.PredictionCreate,
        status=task.status,
        time_post=datetime.now()
    )

    return crud.create_prediction(db, prediction)


@api.get('/result/{task_id}', response_model=schemas.Prediction)
async def get_results(task_id: str, db: Session = Depends(get_db)):
    task = AsyncResult(task_id)
    task_id, status = str(task), task.status

    db_pred = crud.get_prediction(db, task_id)

    if db_pred is None:
        raise HTTPException(status_code=404, detail='Task not found')

    db_pred.status = status

    if task.failed():
        crud.update_prediction(db, db_pred)

        raise HTTPException(status_code=500, detail='Task failed')

    if not task.ready():
        return db_pred
    
    y = task.get()

    db_pred.y = y
    db_pred.time_get = datetime.now()

    return crud.update_prediction(db, db_pred)

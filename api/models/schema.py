from pydantic import BaseModel

class Prediction(BaseModel):
    task_id: str
    x: float
    y: float
    status: str

    class Config:
        orm_mode = True

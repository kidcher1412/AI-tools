from fastapi import FastAPI
from sqlalchemy import create_engine
from datetime import datetime,timedelta
engine = create_engine('sqlite:///DeskTop_App/database.db?check_same_thread=False', echo=True)
connection = engine.connect()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

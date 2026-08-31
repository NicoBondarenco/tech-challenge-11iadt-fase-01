# fastapi dev src/main.py

from fastapi import FastAPI

from src.controller.model_controller import router


app = FastAPI()
app.include_router(router)

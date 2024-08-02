from fastapi import FastAPI
from router import predict_signals

app = FastAPI()

app.include_router(predict_signals.router)

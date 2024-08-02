from fastapi import APIRouter
from model.invoke_prediction import request_signal_prediction
from model.pydantic_signals import SignalsBody

router = APIRouter()


@router.post("/predict_signals/")
def predict_signals(signal_body: SignalsBody):
    response = request_signal_prediction(signal_body)
    return response

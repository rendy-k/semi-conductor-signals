import numpy as np
import pandas as pd
import pickle
from model.pydantic_signals import SignalsBody

# Load model
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))


# Invoke prediction
def invoke_signal_prediction(
    feat_366, feat_561, feat_19, feat_137, feat_562, feat_563, feat_384, feat_385, feat_65, feat_59
):
    X_train = pd.DataFrame(
        [[feat_366, feat_561, feat_19, feat_137, feat_562, feat_563, feat_384, feat_385, feat_65, feat_59]],
        columns = ['366', '561', '19', '137', '562', '563', '384', '385', '65', '59']
    )
    pred = xgb_model.predict_proba(X_train)[:, 1]
    return float(pred[0])


def zero_to_na(x: float):
    return x if x != 0 else np.nan


def request_signal_prediction(signal_body: SignalsBody):   
    result = invoke_signal_prediction(
        zero_to_na(signal_body.signal_1),
        zero_to_na(signal_body.signal_2),
        zero_to_na(signal_body.signal_3),
        zero_to_na(signal_body.signal_4),
        zero_to_na(signal_body.signal_5),
        zero_to_na(signal_body.signal_6),
        zero_to_na(signal_body.signal_7),
        zero_to_na(signal_body.signal_8),
        zero_to_na(signal_body.signal_9),
        zero_to_na(signal_body.signal_10),
    )

    return result

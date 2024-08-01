import numpy as np
import pandas as pd
import pickle

# Load model
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))

# Invoke prediction
def invoke_prediction(
    feat_366, feat_561, feat_19, feat_137, feat_562, feat_563, feat_384, feat_385, feat_65, feat_59
):
    X_train = pd.DataFrame(
        [[feat_366, feat_561, feat_19, feat_137, feat_562, feat_563, feat_384, feat_385, feat_65, feat_59]],
        columns = ['366', '561', '19', '137', '562', '563', '384', '385', '65', '59']
    )
    pred = xgb_model.predict_proba(X_train)[:, 1]
    return pred[0]

import pandas as pd
import pickle


# Load model
xgb_model = pickle.load(open("model/xgb_model.pkl", "rb"))


# Invoke prediction
def invoke_signal_prediction(
    feat_366: float,
    feat_561: float,
    feat_19: float,
    feat_137: float,
    feat_562: float,
    feat_563: float,
    feat_384: float,
    feat_385: float,
    feat_65: float,
    feat_59: float,
) -> float:
    """Predict whether the signals are a Pass or Fail

    Args:
        feat_366 (float): signal number
        feat_561 (float): signal number
        feat_19 (float): signal number
        feat_137 (float): signal number
        feat_562 (float): signal number
        feat_563 (float): signal number
        feat_384 (float): signal number
        feat_385 (float): signal number
        feat_65 (float): signal number
        feat_59 (float): signal number

    Returns:
        float: the binary prediction result
    """
    X_train = pd.DataFrame(
        [[feat_366, feat_561, feat_19, feat_137, feat_562, feat_563, feat_384, feat_385, feat_65, feat_59]],
        columns = ['366', '561', '19', '137', '562', '563', '384', '385', '65', '59']
    )
    pred = xgb_model.predict_proba(X_train)[:, 1]
    return float(pred[0])

import numpy as np
import streamlit as st
import requests
from model.invoke_prediction import invoke_signal_prediction
try:
    from config import SIGNAL_BACKEND
except:
    SIGNAL_BACKEND = None
    print("Backend server not found")

def main():
    st.set_page_config(
        page_title="Semi-conductor Signals", page_icon=":laptop:"
    )
    st.title("💻 Semi-conductor Signals: Pass/Fail Prediction")
    st.markdown(
        """This tool accepts 10 numerical semi-conductor signals and predicts whether it is a PASS or FAIL.
        Originally, there are 5890 signals, but 10 of them are the strongest features."""
    )

    with st.form("form"):
        # Input form
        row_0 = st.columns(3)
        with row_0[0]:
            signal_1 = st.number_input("Signal 1", value=0.0029)
        with row_0[1]:
            signal_2 = st.number_input("Signal 2", value=71.0573)
        with row_0[2]:
            signal_3 = st.number_input("Signal 3", value=12.4722)

        row_1 = st.columns(3)
        with row_1[0]:
            signal_4 = st.number_input("Signal 4", value=55.1)
        with row_1[1]:
            signal_5 = st.number_input("Signal 5", value=248.928)
        with row_1[2]:
            signal_6 = st.number_input("Signal 6", value=0.8157)

        row_2 = st.columns(3)
        with row_2[0]:
            signal_7 = st.number_input("Signal 7", value=0)
        with row_2[1]:
            signal_8 = st.number_input("Signal 8", value=0.0076)
        with row_2[2]:
            signal_9 = st.number_input("Signal 9", value=18.2879)

        row_3 = st.columns(3)
        with row_3[0]:
            signal_10 = st.number_input("Signal 10", value=8.3018)

        button = st.form_submit_button("Predict . . .")


    if button:
        with st.spinner("Predicting . . ."):
            try:
                params = {
                    "signal_1": signal_1,
                    "signal_2": signal_2,
                    "signal_3": signal_3,
                    "signal_4": signal_4,
                    "signal_5": signal_5,
                    "signal_6": signal_6,
                    "signal_7": signal_7,
                    "signal_8": signal_8,
                    "signal_9": signal_9,
                    "signal_10": signal_10,
                }

                # Request API
                use_api = False
                if SIGNAL_BACKEND is not None:
                    try:
                        response = requests.post(f"{SIGNAL_BACKEND}/predict_signals", json=params)
                        response = response.json()
                        use_api = True
                    except:
                        pass

                # If backend is not deployed, use local model
                if use_api == False:
                    for k, v in params.items():
                        if v == 0:
                            params[k] = np.nan

                    response = invoke_signal_prediction(
                        signal_1, signal_2, signal_3, signal_4, signal_5,
                        signal_6, signal_7, signal_8, signal_9, signal_10
                    )

                response = round(response, 2)
                print(use_api)

                # Display result
                st.markdown("# Prediction result")
                pass_fail = "FAIL" if response > 0.33 else "PASS"
                st.markdown(f"Prediction score: {str(response)} ({pass_fail})")

            except Exception as e:
                st.markdown("Error in running the prediction")
                st.exception(e)     


if __name__ == "__main__":
    main()

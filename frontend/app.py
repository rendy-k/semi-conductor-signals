import numpy as np
import streamlit as st
from model.invoke_prediction import invoke_prediction


def main():
    st.set_page_config(
        page_title="Semi-conductor Signals", page_icon=":laptop:"
    )
    st.title("💻 Semi-conductor Signals: Pass/Fail Prediction")

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
        with st.spinner("Computing image segmentation"):
            try:
                inputs = {
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

                for k, v in inputs.items():
                    if v == 0:
                        inputs[k] = np.nan

                # Request API

                # If backend is not deployed, use local model
                response = invoke_prediction(
                    signal_1, signal_2, signal_3, signal_4, signal_5,
                    signal_6, signal_7, signal_8, signal_9, signal_10
                )
                response = round(response, 2)

                # Display result
                st.markdown("# Prediction result")
                pass_fail = "FAIL" if response > 0.5 else "PASS"
                st.markdown(f"Prediction score: {str(response)} ({pass_fail})")

            except Exception as e:
                st.markdown("Error in running the prediction")
                st.exception(e)     


if __name__ == "__main__":
    main()

import numpy as np
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

url = "http://model_backend:8000/predict"

if 'prediction_value' not in st.session_state:
    st.session_state.prediction_value = "0"
    st.session_state.confidence_value = "0"
    st.session_state.label_value = "0"

    st.session_state.history = {
        "timestamp": [],
        "pred": [],
        "label": []
    }


def submit():
    if canvas_result.image_data is not None:
        label = st.session_state.label_value

        image = canvas_result.image_data
        image = np.max(image[:, :, :3], axis=2).clip(0, 1).astype(int)

        data = {"image": image.tolist(), "label": str(label)}

        response = requests.post(url, json=data)
        response_json = response.json()
        prediction_value = response_json["label"]
        confidence = response_json["confidence"]
        history = response_json["history"]

        st.session_state.prediction_value = str(prediction_value)
        st.session_state.confidence_value = str(confidence)
        st.session_state.history = history

        st.rerun()

hide_label_style = """
    <style>
    div[data-testid="stTextInput"] label {display: none;}
    </style>
"""
st.markdown(hide_label_style, unsafe_allow_html=True)

st.title("Digit Recogniser")

canvas_col, input_output = st.columns(2)
with canvas_col:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_color="white",
        stroke_width=20,
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with input_output:
    labels, widgets = st.columns(2)
    with labels:
        st.markdown("Prediction")
        st.markdown("Confidence")
        st.markdown("True Label")
    with widgets:
        st.markdown(st.session_state.prediction_value)
        st.markdown(st.session_state.confidence_value)
        st.session_state.label_value = st.text_input("", value=st.session_state.label_value)

    if st.button("Submit"):
        submit()

st.markdown("History")

st.dataframe(st.session_state.history)

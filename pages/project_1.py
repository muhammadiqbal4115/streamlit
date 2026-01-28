import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import default_rng as rng
import os
from sklearn.naive_bayes import GaussianNB
import joblib

st.set_page_config(
    page_title="MAAN-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Force the tab list container to take 100% width */
    [data-baseweb="tab-list"] {
        width: 100%;
        display: flex;
    }
    .e12zf7d53{
        display: flex;
        justify-content: center;
        align-items: center; }
    #predict-iris-species{
            text-align: center;}
    section[data-testid="stSidebar"] ul {
    visibility: hidden;
    height: 0px;
    }

    /* Force each tab button to grow and fill the container equally */
    [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:

    st.page_link(
        "my_portfolio.py",
        label="Portfolio",
    )

    st.markdown(
        '<div class="sidebar-heading">📁 Projects</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-subheading">Machine Learning:</div>',
        unsafe_allow_html=True,
    )

    st.page_link(
        "pages/project_1.py",
        label="📌 Predict Iris Species",
    )

    st.page_link(
        "pages/project_2.py",
        label="📌 Predict Housing Price",
    )
# Project 1

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(base_dir, "static_files/images", "01_iris.jpg")
file_path = os.path.join(base_dir, "static_files", "01_iris-train.xlsx")
data = pd.read_excel(file_path)

data = pd.read_excel(file_path)
X = data.iloc[:, :-1]  # all columns except last
y = data.iloc[:, -1]  # last column = target

model_path = os.path.join(base_dir, "static_files", "01_iris_model.pkl")

message_placeholder = st.empty()

# Check if model exists
if os.path.exists(model_path):
    # Load existing model
    model = joblib.load(model_path)
else:
    # Train and save model
    model = GaussianNB()
    model.fit(X.values, y.values)  # type: ignore
    joblib.dump(model, model_path)

with st.container():
    st.image(image_path, width=200)
with st.container():
    st.title("Predict Iris Species")


with st.container():
    with st.form("iris_form"):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                sepal_length = st.number_input(
                    "Sepal Length (cm)",
                    min_value=4.0,
                    max_value=8.0,
                    step=0.1,
                    value=4.0,  # default is 0
                )
            with col2:
                sepal_width = st.number_input(
                    "Sepal Width (cm)",
                    min_value=2.0,
                    max_value=4.5,
                    step=0.1,
                    value=2.0,
                )

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                petal_length = st.number_input(
                    "Petal Length (cm)",
                    min_value=1.0,
                    max_value=7.0,
                    step=0.1,
                    value=1.0,
                )
            with col2:
                petal_width = st.number_input(
                    "Petal Width (cm)",
                    min_value=0.1,
                    max_value=2.5,
                    step=0.1,
                    value=0.1,
                )
        submit = st.form_submit_button("Predict Iris Species")
        if submit:
            features = np.array(
                [[sepal_length, sepal_width, petal_length, petal_width]]
            )

            if sepal_length == 0.0 or sepal_width == 0.0 or petal_length == 0.0 or petal_width == 0.0:
                pass
            else:
                prediction = model.predict(features)
                st.success(f"Predicted class: {prediction[0]}")

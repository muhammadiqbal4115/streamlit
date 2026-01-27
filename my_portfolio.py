import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import default_rng as rng

st.set_page_config(
    page_title="MAAN-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* Force the tab list container to take 100% width */
    [data-baseweb="tab-list"] {
        width: 100%;
        display: flex;
    }
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
    """, unsafe_allow_html=True)

with st.sidebar:

    st.page_link(
        "my_portfolio.py",
        label="Portfolio",
    )

    st.markdown('<div class="sidebar-heading">📁 Projects</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subheading">Machine Learning:</div>', unsafe_allow_html=True)

    st.page_link(
        "pages/project_1.py",
        label="📌 Predict Iris Species",
    )

    st.page_link(
        "pages/project_2.py",
        label="📌 Predict Housing Price",
    )

tab1, tab2 = st.tabs(["Profile", "Initial Streamlit"], width="stretch")

with tab1:
    with st.container():
        col1, col2 = st.columns(2)
        with st.container():
            with col1:
                st.title('')
                current_dir = Path(__file__).parent
                img_path = current_dir / "1758954174262.jpg"
                st.image(img_path, width="stretch")
        with st.container():
            with col2:
                st.title('')
                st.subheader("Name: Muhammad Iqbal")
                st.subheader('Email: muhammad.iqbal4115@gmail.com')
                st.subheader('City: Faisalabad')
with tab2:
    st.title("Title:")
    st.code('''st.title("Title:")''')
    st.divider()

    st.write("Write:")
    st.code('''st.write("Write:")''')
    st.divider()

    st.header("Header:")
    st.code('''st.header("Header:")''')
    st.divider()

    st.subheader("SubHeader:")
    st.code('''st.subheader("SubHeader:")''')
    st.divider()

    st.header("Markdown:")
    st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
    st.code("""st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')""")
    st.divider()

    st.header('Text')
    st.text("This is text\n[and more text](that's not a Markdown link).")
    st.code('''st.text("This is text\n[and more text](that's not a Markdown link).")''')
    st.divider()

    st.header('DataFrame:')
    df = pd.DataFrame(np.random.randn(50, 20), columns=[f'col-{i}' for i in range(20)])
    st.dataframe(df)
    st.code("""df = pd.DataFrame(np.random.randn(50, 20), columns=[f'col-{i}' for i in range(20)])
    st.dataframe(df)""")
    st.divider()

    st.header('Bar Chart')
    df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
    st.bar_chart(df)
    st.code("""df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
    st.bar_chart(df)""")
    st.divider()

    st.header('Map:')
    st.map()
    st.code("""st.map()""")


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
        label="📌 Iris Data-Set",
    )

    st.page_link(
        "pages/project_2.py",
        label="📌 Project 2",
    )
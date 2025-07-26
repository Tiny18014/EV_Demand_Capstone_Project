from datetime import datetime, timedelta
import pickle
from river import metrics
from src.data.preprocess import preprocess  
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

st.set_page_config(layout="wide", page_title="Sentiment Forecast Dashboard", page_icon="📈")
st.markdown(
    """
    <style>
    /* Narrower sidebar */
    .stApp {
        background-color: #4a5a34;
        color: #ffffff
    }
    header[data-testid="stHeader"] {
        background-color: #4a5a34; /* 🔥 Change this to whatever you want */
        color: #000000;
    }
    section[data-testid="stSidebar"] {
        width: 20px !important;
        background-color: #2e3d2a
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
        padding-top: 10px;
        align-items: center;
    }

    /* Center the icons */
    .css-1aumxhk {
        justify-content: center;
    }

    /* Optional: remove Streamlit hamburger menu and footer */
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    selected = option_menu(
        None,
        ["Home", "Pulse", "Vibe", "Charge"],
        icons=["house", "graph-up-arrow", "speedometer", "car-front-fill"],
        menu_icon=None,
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "background-color": "#2e3d2a",
                "padding": "0!important",
            },
            "icon": {"color": "#faf7e6", "font-size": "20px"},
            "nav-link": {
                "text-align": "center",
                "margin": "0 auto",
                "padding": "10px 0",
                "background-color": "#2e3d2a",
                "color": "#faf7e6"
            },
            "nav-link-selected": {
                "background-color": "#2e3d2a",
            },
        }
    )
st.sidebar.markdown("---")
st.sidebar.markdown("**📂 [GitHub Repo](https://github.com/your/repo)**")
st.sidebar.markdown("**🔄 Version:** `v1.0.0`")
st.sidebar.markdown("**👩‍💻 Built by:** [@you](https://github.com/you)")

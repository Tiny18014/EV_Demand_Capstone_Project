from datetime import datetime, timedelta
import pickle
from river import metrics
from src.data.preprocess import preprocess  
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

st.set_page_config(page_title='EV', page_icon='🗣️', layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #0D1B2A;
            color: #E0E6ED;
        }
        .stTextArea textarea {
            background-color: #f2f2f2;
            color: #006064;
        }
        .stButton button {
            background-color: #f5cb5c;
            color: #0D1B2A;
        }
        .stButton button:hover {
            background-color: #f5cb5c; /* Button hover background color */
            color: #0d1b2a; /* Button hover text color */
        }
        .prediction-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 20px;
            margin: 20px 0;
        }
        .not-offensive {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .offensive {
            background-color: #ffcdd2;
            color: #b71c1c;
        }
    </style>
""", unsafe_allow_html=True)

# Embed the logo in the HTML
st.markdown(f"""
    <div style="text-align: center;">
        <h1>EV Analysis</h1>
        <h4><i>Understand your EV market</i></h4>
    </div>
    """, unsafe_allow_html=True)
st.write("")

st.markdown("<p style='text-align: center;'>This hate speech detection application helps make online spaces safer by identifying and mitigating harmful content in text, audio, and video. This application aims to promote a healthier, more respectful online environment.</p>", unsafe_allow_html=True)
st.write("")
st.write("")

cols = st.columns([1, 2, 1, 2, 1, 2, 1, 2, 1])

# Place buttons in every other column
with cols[1]:
    if st.button("Cognition"):
        st.session_state.input_type = "sentiment"
with cols[3]:
    if st.button("Projections"):
        st.session_state.input_type = "demand"
with cols[5]:
    if st.button("Dynamics"):
        st.session_state.input_type = "charge"
with cols[7]:
    if st.button("Synopsis"):
        st.session_state.input_type = "about"


# Display the relevant input fields and outputs based on the selected input type
if 'input_type' in st.session_state:
    if st.session_state.input_type == "sentiment":
        st.header('Sentiment Analysis and Forecasting')


    elif st.session_state.input_type == "demand":
        st.header('Sales Forecasting')

    elif st.session_state.input_type == "charge":
        st.header('Charging Behavior and Energy Consumption Analysis')
          
    elif st.session_state.input_type == "about":
        st.header('What is hate speech?')
        st.markdown("Hate speech is communication that attacks a person or group on the basis of attributes such as race, religion, ethnic origin, national origin, disability, or gender identity. It can use offensive language, promote violence, or spread negative stereotypes. Hate speech can be online or offline, and is spread across various media.")
        st.header('About')
        st.markdown('This machine learning application incorporates the Hate Speech Dataset from Kaggle, and has the primary functionality of classifying <b>text, audio</b> and <b>video</b> as hate speech, offensive language, and non offensive language. This project is primarily a <b>Natural Language Processing</b> application, aimed at extracting the context from various media to further classify as hate speech. <b>Extended Gradient Boosting</b> model is used for the final predictions and <b>Term Frequency - Inverse Document Frequency</b> vectorizer is used to vectorize and convert the words to numerical values. The libraries used for audio and video conversion are <b>pydub</b> and <b>moviepy</b> respectively. Audio is first transcribed into text, and then predicted. Video is converted to audio and then transcribed to text. This project is developed using <b>Streamlit</b>.', unsafe_allow_html=True)


#disclaimer and footer
st.markdown(" ")
st.markdown("<h2 align=center> #SayNoToHate<h2>", unsafe_allow_html=True)
st.markdown(" ")
st.markdown(
    """
    <div style="background-color: #00bfa6;
            color: #000000; border: 1px solid #00bfa6; padding: 10px; border-radius: 20px;">
        <h6 style='text-align: center;'>Disclaimer</h6>
        <p style='text-align: center;'>
            This application is designed to detect and analyze instances of hate speech in text, audio, and video. 
            During its operation, offensive language may be displayed, thus user discretion is advised.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown("<p style='text-align: center; color: #ffffff;'>© 2024 HateShield. All rights reserved.</p>", unsafe_allow_html=True)

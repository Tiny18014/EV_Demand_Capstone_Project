from datetime import datetime, timedelta
import pickle
from river import metrics
from src.data.preprocess import preprocess  
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from src.model.agent_sentiment import tech_to_business_agent

st.set_page_config(page_title='EV', page_icon='🗣️', layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #112235;
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
        brand_decoder = {
            0: 'Ashok Leyland',
            1: 'Hero MotoCorp',
            2: 'Mahindra',
            3: 'Tata Motors',
            4: 'TVS Motors'
        }
        brands = ['Ashok Leyland', 'Hero MotoCorp', 'Mahindra', 'Tata Motors', 'TVS Motors']
        #Getting datsets and models ready
        df1 = pd.read_csv("src/data/feat.csv")
        dff = pd.read_csv("src/data/targ.csv")
        yy = dff['sentiment_label']
        df = pd.concat([df1, dff], axis=1)
        df['index'] = df.index
        df['brand_name'] = df['brand_encoded'].map(brand_decoder).fillna('Unknown Brand')
        avg_sentiment_vader = df.groupby("brand_name")["vader_roll_avg"].mean().reset_index(name="avg_vader")
        avg_sentiment_bert = df.groupby("brand_name")["bert_roll_avg"].mean().reset_index(name="avg_bert")
        avg_sentiment = pd.merge(avg_sentiment_vader, avg_sentiment_bert, on="brand_name")

        # Melt so both sentiment columns are in one column
        avg_sentiment_melted = avg_sentiment.melt(
            id_vars="brand_name", 
            value_vars=["avg_vader", "avg_bert"], 
            var_name="sentiment_type", 
            value_name="avg_sentiment"
        )

        df["buzz_score"] = df["news_volume"] * df["vader_compound"]
        buzz_scores = df.groupby("brand_encoded")["buzz_score"].sum().reset_index(name="buzz_score")
        corr = df[['vader_compound', 'bert_roll_avg', 'trend_delta', 'news_volume']].corr()
        fig_momentum = px.line(df, x="index", y="vader_momentum", color="brand_name", markers=True)

    # Bar chart: average sentiment
        fig_bar_sentiment = px.bar(
            avg_sentiment_melted,
            x="brand_name",
            y="avg_sentiment",
            color="sentiment_type",
            barmode="group"
        )

        fig_buzz = px.bar(buzz_scores, x="brand_encoded", y="buzz_score", title="Buzz Score per Brand")
        fig_buzz.update_xaxes(tickvals=list(brand_decoder.keys()), ticktext=list(brand_decoder.values()))
        agg_funcs = {
        'vader_compound': 'mean',
        'vader_momentum': 'mean',
        'vader_roll_avg': 'mean',
        'sentiment_stddev': 'mean',
        'news_volume': 'sum',
        'trend_delta': 'mean',
        'bert_roll_avg': 'mean',
        'sentiment_label_lag1': 'mean',
        'bert_score_lag1': 'mean',
        'vader_lag1': 'mean',
        'vader_trend_interaction': 'mean',
        'volume_volatility_interaction': 'mean',
        'transition_count_last_3_days': 'mean',
        'sentiment_label': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        }
        
        brand_summary = df.groupby('brand_name').agg(agg_funcs).reset_index()
        with open("src/model/rm_uptoaugustnow.pkl", "rb") as f:
            model = pickle.load(f)
        #Actual Streamlit Code
        st.header('Sentiment Analysis and Forecasting')
        st.markdown(
    "<p style='color: #9BAEC1;'>How did the sentiment vary over the past week? How's it going to change for the next week? We've got the analysis.</p>", 
    unsafe_allow_html=True
)

        cols_pred = st.columns(5)
        from river import metrics  # Overall class-wise performance
        accuracy = metrics.Accuracy()
        f1 = metrics.F1()
        precision = metrics.Precision()
        recall = metrics.Recall()
        for xi, yi in zip(df1.to_dict(orient='records'), yy):
            y_pred = model.predict_one(xi)
            accuracy.update(yi, y_pred)
            f1.update(yi, y_pred)
            precision.update(yi, y_pred)
            recall.update(yi, y_pred)

        for i, brand_id in enumerate(range(5)):
            x_brand = df1[df1['brand_encoded'] == brand_id]

            if x_brand.empty or len(x_brand) < 2:
                with cols_pred[i]:
                    st.warning(f"⚠️ No data for **{brands[brand_id]}**.")
                continue

            # Latest and previous prediction
            last_xi = x_brand.iloc[-1].to_dict()
            y_pred = model.predict_one(last_xi)

            prev_xi = x_brand.iloc[-2].to_dict()
            y_prev = model.predict_one(prev_xi)

            # Determine base and hover colors, arrow, and sentiment
            if y_pred == 1.0:
                base_bg = "rgba(168, 230, 207, 0.25)"
                hover_bg = "rgba(168, 230, 207, 0.5)"
                base_text = "#9BAEC1"
                hover_text = "#112235"
                arrow = "↗"
                arrow_color = "#124628"
                sentiment_text = "Positive sentiment predicted next week"
            else:
                base_bg = "rgba(255, 99, 99, 0.25)"
                hover_bg = "rgba(255, 99, 99, 0.5)"
                base_text = "#9BAEC1"
                hover_text = "#112235"
                arrow = "↘"
                arrow_color = "#681313"
                sentiment_text = "Negative sentiment predicted next week"

            # Add CSS for hover effect
            st.markdown(f"""
                <style>
                .hover-box-{i} {{
                    background: {hover_bg};
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding:15px;
                    text-align: center;
                    color: {hover_text};
                    transition: background 0.3s ease, color 0.3s ease;
                }}
                </style>
            """, unsafe_allow_html=True)

            # Render box with diagonal arrow
            with cols_pred[i]:
                st.markdown(f"""
                    <div class="hover-box-{i}">
                        <h5 style='color: inherit; text-align: center;'>{brands[brand_id]}</h5>
                        <h1 style='font-size: 100px; color:{arrow_color};'>{arrow}</h1>
                    </div>
                """, unsafe_allow_html=True)

        #graphs from plotly
        custom_colors = ['#00FFC6', '#0077B6',"#48CAE4", '#90E0EF', '#FFD166']
        background_color = "#112235"
        font_color = '#9BAEC1'  # or try '#e0e0e0' for softer look

    # Apply to line charts
        for fig in [fig_momentum]:
            fig.update_layout(
                paper_bgcolor=background_color,
                plot_bgcolor=background_color,
                font=dict(color=font_color),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                colorway=custom_colors
            )
            fig.update_xaxes(showgrid=True, gridcolor='gray')
            fig.update_yaxes(showgrid=True, gridcolor='gray')

        # Apply to bar charts
        for fig in [fig_bar_sentiment, fig_buzz]:
            fig.update_traces(marker_color=custom_colors)
            fig.update_layout(
                paper_bgcolor=background_color,
                plot_bgcolor=background_color,
                font=dict(color=font_color),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='gray')
            )
        
        c1_graph, c2_graph = st.columns(2)
        with c1_graph:

            st.subheader("Sentiment Momentum Comparison")
            st.plotly_chart(fig_momentum, use_container_width=True)
            st.markdown(
    "<h6 style='text-align: center; color: #F5CB5C;'>Shows sentiment momentum over time for each brand.</h4>", 
    unsafe_allow_html=True
)
        with c2_graph:

            st.subheader("Sentiment Intensity Comparison")
            st.plotly_chart(fig_bar_sentiment, use_container_width=True)
            st.markdown(
    "<h6 style='text-align: center; color: #F5CB5C;'>Compares overall sentiment intensity across brands.</h4>", 
    unsafe_allow_html=True
)

        #agent and model metrics
        c1,c2 = st.columns([2,1])
        with c1:
            st.subheader("Brand-Level Analysis")
            progress = st.progress(0)
            insights = []

            """for i, (_, row) in enumerate(brand_summary.iterrows()):
                result = tech_to_business_agent.run(str(row.to_dict()))
                insights.append(result.content)
                progress.progress((i + 1) / len(brand_summary))
            st.success("Analysis complete ✅")
            for insight in insights:
                with st.expander(insight.split('\n')[0].strip('#').strip(), expanded=False):
                    st.markdown(insight, unsafe_allow_html=False)"""
        with c2:
            st.subheader("Model Metrics")
            st.metric(label="Accuracy", value=f"{accuracy.get():.2%}")
            st.metric(label="F1 Score", value=f"{f1.get():.2f}")
            st.metric(label="Precision", value=f"{precision.get():.2f}")
            st.metric(label="Recall", value=f"{recall.get():.2f}")


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
st.markdown("<p style='text-align: center; color: #ffffff;'>© 2025 EVAnalysis. All rights reserved.</p>", unsafe_allow_html=True)

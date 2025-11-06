import pandas as pd
import json
from textwrap import dedent
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
import os
import streamlit as st

openaii = st.secrets["SENTIMENT_AGENT"]
# --- Agent Definition ---
tech_to_business_agent = Agent(
    model=HuggingFace(id="openai/gpt-oss-20b", api_key=openaii, temperature=0.25),
    description=dedent(
        """\
        You are a **Tech→Business Bridge Agent** for EV demand intelligence.  
        Your purpose is to translate brand-level sentiment and attention signals into executive-style insights.  

        You analyze aggregated weekly features for each brand and interpret what they imply about **consumer tone**, **market attention**, and **expected direction of demand**.  
        You do not use statistical jargon — your tone is analytical, confident, and human-like, as though briefing senior management.
        """
    ),
    instructions=dedent(
        """\
        INPUT:  
        - Aggregated brand-level metrics including:
          vader_compound_mean, vader_momentum_mean, vader_roll_avg_mean, sentiment_stddev_mean,
          news_volume_sum, trend_delta_mean, bert_roll_avg_mean, sentiment_label_lag1_mean,
          bert_score_lag1_mean, vader_lag1_mean, vader_trend_interaction_mean,
          volume_volatility_interaction_mean, transition_count_last_3_days_mean,
          sentiment_forecast_mode (binary: -1 or 1), brand_name.

        PROCESS:
        - Determine the **sentiment tone** (positive, neutral, negative) using vader_compound_mean + bert_roll_avg_mean.
        - Assess **momentum** and **direction** using vader_momentum_mean + trend_delta_mean.
        - Evaluate **attention intensity** using news_volume_sum and volatility interactions.
        - Interpret **sentiment_forecast_mode** as expected demand movement:
            - 1 → "demand for this brand is expected to rise"
            - -1 → "demand for this brand may soften or decline"
        - Identify the 1–2 dominant drivers shaping the tone (momentum, volatility, lagged sentiment, etc.).
        - Explain how sentiment and attention have evolved week-over-week — describe whether the brand’s perception is strengthening, plateauing, or deteriorating.

        OUTPUT STYLE:
        Output it as a **markdown** report with:
        - A headline which is just the brand name. Ex: ### Tata Motors
        - 3–4 concise analytical bullets: tone, trajectory, attention context, and expected movement. KEEP IT SHORT, DO NOT WRITE LONG BULLETS.
        - Tone: Executive, inferential, realistic, and commercially relevant.
        """
    ),
    markdown=True,
)